"""One-token routed NVFP4 decode using the existing lane device-table dispatch."""

from __future__ import annotations

import math

import torch

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.families.glm5_next.cuda import glue, nvfp4_dispatch
from tensorfold.families.glm5_next.cuda.nvfp4_table import PROJECTIONS, RoutedTable


def routed_decode(x: torch.Tensor, table: RoutedTable, ids: torch.Tensor, weights: torch.Tensor,
                  out: torch.Tensor, *, residual_act: float, intermediate_act: float,
                  limit: float = 10.0) -> torch.Tensor:
    """Write a single token's routed plus shared MLP result into caller-owned fp32 ``out``.

    ``ids`` contains routed ids only (normally eight); the BF16 shared branch is
    appended with weight one. The explicit static input factors apply to every
    gate/up and down respectively: table alpha slots hold output factors only.
    Each selected slot gets its own intermediate quantization, even at weight
    zero. Projection weights, dimensions and alphas come from CUDA table slots;
    this path NEVER accesses ``table.linears``. The caller MUST retain the packed
    storage owners through CUDA completion, including when clearing that list.

    This is eager lane execution: slot validation/dispatch synchronizes. Refused
    destinations raise before any quantization or MLP work; no replacement buffer
    is returned. Storage overlap is conservatively refused, including disjoint
    destination views into source storage.
    """
    selected, width, hidden = _validate(x, table, ids, weights, out, residual_act, intermediate_act, limit)
    residual = checkpoint.quant4(x, residual_act)
    slots = len(selected)
    outputs = torch.empty((1, slots + 1, hidden), dtype=torch.float32, device=x.device)
    for slot, expert in enumerate(selected):
        gate = torch.empty((1, width), dtype=torch.bfloat16, device=x.device)
        up = torch.empty_like(gate)
        nvfp4_dispatch.projection(residual, table, expert, nvfp4_dispatch.GATE, gate)
        nvfp4_dispatch.projection(residual, table, expert, nvfp4_dispatch.UP, up)
        activated = _swiglu(torch.cat((gate, up), dim=1), width, limit)
        intermediate = checkpoint.quant4(activated, intermediate_act)
        nvfp4_dispatch.projection(intermediate, table, expert, nvfp4_dispatch.DOWN,
                                  outputs[:, slot, :])

    shared = table.shared
    gu = torch.cat((torch.nn.functional.linear(x, shared["gate_proj"]),
                    torch.nn.functional.linear(x, shared["up_proj"])), dim=1)
    activated = _swiglu(gu, width, limit)
    outputs[:, slots, :] = torch.nn.functional.linear(activated.float(), shared["down_proj"].float())
    combined = torch.empty((1, slots + 1), dtype=torch.float32, device=x.device)
    combined[:, :slots] = weights
    combined[:, slots] = 1.0
    glue.combine(outputs, combined, out)
    return out


def _swiglu(gu: torch.Tensor, width: int, limit: float) -> torch.Tensor:
    activated = torch.empty((1, width), dtype=torch.bfloat16, device=gu.device)
    sums = torch.empty((1, width // 64), dtype=torch.float32, device=gu.device)
    glue.swiglu(gu, activated, sums, limit)
    return activated


def _storage(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def _validate(x, table, ids, weights, out, residual_act, intermediate_act, limit):
    if (x.ndim != 2 or x.shape[0] != 1 or x.shape[1] <= 0 or x.shape[1] % 64 or
            not x.is_contiguous() or not x.is_cuda or x.dtype != torch.bfloat16):
        raise ValueError("decode input must be contiguous CUDA bf16 with shape (1, K)")
    for name, value in (("residual_act", residual_act), ("intermediate_act", intermediate_act), ("limit", limit)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(table, RoutedTable):
        raise TypeError("expected RoutedTable")
    if table.words_ptr.ndim != 2 or table.words_ptr.shape[0] <= 0 or table.words_ptr.shape[1] != 3:
        raise ValueError("tables must have shape (positive experts, 3)")
    experts = table.words_ptr.shape[0]
    fields = ((table.words_ptr, torch.int64), (table.bs_ptr, torch.int64),
              (table.n, torch.int32), (table.k, torch.int32), (table.alpha, torch.float32))
    if any(t.dtype != dtype or t.device != x.device or tuple(t.shape) != (experts, 3) or
           not t.is_contiguous() for t, dtype in fields):
        raise ValueError("tables must be contiguous CUDA (experts, 3) with int64 pointers, int32 n/k, fp32 alpha")
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] <= 0 or ids.dtype != torch.int64 or ids.device != x.device:
        raise ValueError("ids must be CUDA int64 with shape (1, positive slots)")
    selected = ids[0].tolist()
    if any(expert < 0 or expert >= experts for expert in selected):
        raise ValueError("expert id is out of range; the shared slot is not an NVFP4 id")
    if weights.shape != ids.shape or weights.dtype != torch.float32 or weights.device != x.device:
        raise ValueError("weights must be fp32 on the input device with the same shape as ids")
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("routed weights must be finite")

    ns, ks = table.n.tolist(), table.k.tolist()
    width, hidden = ns[selected[0]][0], ns[selected[0]][2]
    block = min(1024, hidden)
    if width <= 0 or width % 64 or hidden <= 0 or hidden % block or block & (block - 1):
        raise ValueError("width must be a positive multiple of 64; hidden must fit whole glue.combine blocks")
    for n, k in zip(ns, ks):
        if n != [width, width, hidden] or k != [x.shape[1], x.shape[1], width] or x.shape[1] % 64:
            raise ValueError("expert gate/up/down dimensions must match input, intermediate, and output widths")
    if not bool(torch.isfinite(table.alpha).all()):
        raise ValueError("table alpha must be finite")
    wp, bp = table.words_ptr.tolist(), table.bs_ptr.tolist()
    if any(ptr <= 0 or ptr % 16 for row in wp + bp for ptr in row):
        raise ValueError("projection addresses must be nonzero and aligned")
    shared_sources = []
    for name, shape in zip(PROJECTIONS, ((width, x.shape[1]), (width, x.shape[1]), (hidden, width))):
        weight = table.shared.get(name)
        if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16 or weight.device != x.device or
                tuple(weight.shape) != shape or not weight.is_contiguous()):
            raise ValueError(f"shared expert {name} must be contiguous CUDA bf16 with shape {shape}")
        shared_sources.append(weight)
    if out.shape != (1, hidden):
        raise ValueError("destination must have shape (1, hidden)")
    if out.dtype != torch.float32 or out.device != x.device:
        raise ValueError("destination must be fp32 on the input device")
    if not out.is_contiguous():
        raise ValueError("destination must be contiguous and have no internal overlap")
    spans = [_storage(t) for t in [x, ids, weights, *shared_sources, *(t for t, _ in fields)]]
    for expert in range(experts):
        for col in range(3):
            npad = (ns[expert][col] + 63) // 64 * 64
            spans.extend(((wp[expert][col], npad * ks[expert][col] // 2),
                          (bp[expert][col], npad * ks[expert][col] // 16)))
    start, size = _storage(out)
    if any(start < ptr + length and ptr < start + size for ptr, length in spans):
        raise ValueError("destination must not overlap input, routing, table, shared, or projection storage")
    return selected, width, hidden
