"""Dense and routed NVFP4 eager oracles for the GLM5-next MLP.

Gate and up share input quantization only for matching activation factors.
The activation is the CUDA glue SwiGLU. Lane and prompt are different
reductions and are not compared bitwise. Expert downs stay distinct.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear
from tensorfold.families.glm5_next.cuda import glue


def dense_eager(x: torch.Tensor, gate: Fp4Linear, up: Fp4Linear, down: Fp4Linear,
                out: torch.Tensor, *, limit: float = 10.0) -> torch.Tensor:
    """Evaluate a dense gate/up/down MLP into ``out``.

    Gate/up use the lane backend with one common quantization and separate weight
    factors. GLM's clamped, bf16-rounded SwiGLU feeds one fp32 lane down.
    ``out`` MUST be contiguous fp32 with shape (rows, down.n). Destinations that
    share storage with the input or weights are conservatively refused, including
    disjoint views. Every refusal happens before quantization or multiplication.
    """
    _refuse_dense(x, gate, up, down, out, limit)
    return _dense_body(x, gate, up, down, out, limit, prompt_rows=False)


def dense_prompt(x: torch.Tensor, gate: Fp4Linear, up: Fp4Linear, down: Fp4Linear,
                 out: torch.Tensor, *, limit: float = 10.0) -> torch.Tensor:
    """Dense eager oracle on the prompt GEMM, not the lane kernel.

    Same refusals as ``dense_eager``. Gate and up share one prompt quantization.
    The down is ``checkpoint.prompt`` into the caller fp32 buffer. This is not
    ``mlp_prompt`` and it is not bitwise with the lane oracle.
    """
    _refuse_dense(x, gate, up, down, out, limit)
    return _dense_body(x, gate, up, down, out, limit, prompt_rows=True)


def routed_eager(x: torch.Tensor, experts: Sequence[tuple[Fp4Linear, Fp4Linear, Fp4Linear]],
                 shared: Mapping[str, torch.Tensor], ids: torch.Tensor, weights: torch.Tensor,
                 out: torch.Tensor, *, backend: str, limit: float = 10.0) -> torch.Tensor:
    """One routed layer into ``out``, plus the BF16 shared expert at weight 1.

    Each assignment is one row and one expert. That expert's gate and up may
    share input quantization. Its down reads only that SwiGLU output. Experts
    may have unequal codes and unequal ``weight_scale_2``. The shared expert is
    not an NVFP4 id. ``backend`` is ``lane`` or ``prompt``; those results are
    not a bitwise pair.
    """
    _refuse_routed(x, experts, shared, ids, weights, out, backend, limit)
    rows, slots = ids.shape
    hidden = out.shape[1]
    ey = torch.empty((rows, slots + 1, hidden), dtype=torch.float32, device=x.device)
    for row in range(rows):
        one = x[row:row + 1]
        for slot in range(slots):
            gate, up, down = experts[int(ids[row, slot])]
            ey[row, slot] = _assignment(one, gate, up, down, backend, limit)
    ey[:, slots] = _shared_mlp(x, shared, limit)
    combined = torch.empty((rows, slots + 1), dtype=torch.float32, device=x.device)
    combined[:, :slots] = weights
    combined[:, slots] = 1.0
    glue.combine(ey, combined, out)
    return out


def _refuse_dense(x: torch.Tensor, gate: Fp4Linear, up: Fp4Linear, down: Fp4Linear,
                  out: torch.Tensor, limit: float) -> None:
    for name, lin in (("gate", gate), ("up", up), ("down", down)):
        if not isinstance(lin, Fp4Linear):
            raise TypeError(f"{name}: expected Fp4Linear")
        if lin.act is None or not math.isfinite(lin.act) or lin.act <= 0:
            raise ValueError(f"{name}: activation factor must be finite and positive")
        if lin.n <= 0 or lin.k <= 0 or lin.k % 64:
            raise ValueError(f"{name}: positive dimensions and K a multiple of 64 required")
    if gate.act != up.act:
        raise ValueError("gate and up activation factors must match for common-input quantization")
    if gate.k != up.k or gate.n != up.n or down.k != gate.n or gate.n % 64:
        raise ValueError("gate/up dimensions must match, with a multiple-of-64 width equal to down.K")
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != gate.k:
        raise ValueError("input must have shape (positive rows, gate.K)")
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError("input must be CUDA bf16")
    if out.shape != (x.shape[0], down.n):
        raise ValueError("destination must have shape (rows, down.n)")
    if out.dtype != torch.float32 or out.device != x.device:
        raise ValueError("destination must be fp32 on the input device")
    if not out.is_contiguous():
        raise ValueError("destination must be contiguous and have no internal overlap")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("SwiGLU limit must be finite and positive")
    sources = [x]
    for lin in (gate, up, down):
        if lin.words.device != x.device or lin.bs.device != x.device:
            raise ValueError("projection weights must be on the input device")
        sources.extend((lin.words, lin.bs))
    if any(out.untyped_storage().data_ptr() == source.untyped_storage().data_ptr() for source in sources):
        raise ValueError("destination must not overlap input or projection storage")


def _dense_body(x: torch.Tensor, gate: Fp4Linear, up: Fp4Linear, down: Fp4Linear,
                out: torch.Tensor, limit: float, *, prompt_rows: bool) -> torch.Tensor:
    gate_up = checkpoint.matmul_group(x, [gate, up], prompt_rows=prompt_rows)
    if gate_up is None:
        raise ValueError("gate/up cannot share input quantization")
    activated = _swiglu(torch.cat(gate_up, dim=1), x.shape[0], gate.n, x.device, limit)
    if prompt_rows:
        return checkpoint.prompt(checkpoint.A4, activated, down, out=out, f32=True)
    return checkpoint.matmul(checkpoint.A4, activated, down, out=out, f32=True)


def _swiglu(gu: torch.Tensor, rows: int, width: int, device, limit: float) -> torch.Tensor:
    activated = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    sums = torch.empty((rows, width // 64), dtype=torch.float32, device=device)
    glue.swiglu(gu, activated, sums, limit)
    return activated


def _assignment(row: torch.Tensor, gate: Fp4Linear, up: Fp4Linear, down: Fp4Linear,
                backend: str, limit: float) -> torch.Tensor:
    prompt_rows = backend == "prompt"
    gate_up = checkpoint.matmul_group(row, [gate, up], prompt_rows=prompt_rows)
    if gate_up is None:
        raise ValueError("gate/up cannot share input quantization")
    activated = _swiglu(torch.cat(gate_up, dim=1), 1, gate.n, row.device, limit)
    slot = torch.empty((1, down.n), dtype=torch.float32, device=row.device)
    if prompt_rows:
        written = checkpoint.prompt(checkpoint.A4, activated, down, out=slot, f32=True)
    else:
        written = checkpoint.matmul(checkpoint.A4, activated, down, out=slot, f32=True)
    if written is not slot:
        slot.copy_(written)
    return slot[0]


def _shared_mlp(x: torch.Tensor, shared: Mapping[str, torch.Tensor], limit: float) -> torch.Tensor:
    gate_w, up_w, down_w = shared["gate_proj"], shared["up_proj"], shared["down_proj"]
    gu = torch.cat((torch.nn.functional.linear(x, gate_w), torch.nn.functional.linear(x, up_w)), dim=1)
    activated = _swiglu(gu, x.shape[0], gate_w.shape[0], x.device, limit)
    return torch.nn.functional.linear(activated.float(), down_w.float())


def _refuse_routed(x: torch.Tensor, experts: Sequence[tuple[Fp4Linear, Fp4Linear, Fp4Linear]],
                   shared: Mapping[str, torch.Tensor], ids: torch.Tensor, weights: torch.Tensor,
                   out: torch.Tensor, backend: str, limit: float) -> None:
    if backend not in ("lane", "prompt"):
        raise ValueError("backend must be lane or prompt")
    if not experts:
        raise ValueError("routed layer needs at least one expert")
    first = experts[0]
    if len(first) != 3 or any(not isinstance(lin, Fp4Linear) for lin in first):
        raise TypeError("each expert must be (gate, up, down) Fp4Linear")
    gate0, up0, down0 = first
    for index, triple in enumerate(experts):
        if len(triple) != 3:
            raise TypeError(f"expert {index} must be (gate, up, down)")
        gate, up, down = triple
        for name, lin in (("gate", gate), ("up", up), ("down", down)):
            if not isinstance(lin, Fp4Linear):
                raise TypeError(f"expert {index} {name}: expected Fp4Linear")
            if lin.act is None or not math.isfinite(lin.act) or lin.act <= 0:
                raise ValueError(f"expert {index} {name}: activation factor must be finite and positive")
            if not math.isfinite(lin.scale):
                raise ValueError(f"expert {index} {name}: weight scale must be finite")
            if lin.words.device != gate0.words.device or lin.bs.device != gate0.words.device:
                raise ValueError("projection weights must be on the input device")
        if gate.act != up.act:
            raise ValueError("gate and up activation factors must match for common-input quantization")
        if (gate.k, gate.n, down.k, down.n) != (gate0.k, gate0.n, down0.k, down0.n):
            raise ValueError("expert shapes must match; unequal codes and scales are allowed")
        if gate.k != up.k or gate.n != up.n or down.k != gate.n or gate.n % 64 or gate.k % 64:
            raise ValueError("gate/up dimensions must match, with a multiple-of-64 width equal to down.K")
    if not isinstance(shared, Mapping) or any(key not in shared for key in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError("shared expert needs BF16 gate_proj, up_proj, and down_proj")
    gate_w, up_w, down_w = shared["gate_proj"], shared["up_proj"], shared["down_proj"]
    for name, weight, shape in (
        ("gate_proj", gate_w, (gate0.n, gate0.k)),
        ("up_proj", up_w, (gate0.n, gate0.k)),
        ("down_proj", down_w, (down0.n, gate0.n)),
    ):
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16 or not weight.is_cuda:
            raise ValueError(f"shared expert {name} must be CUDA bf16")
        if weight.device != gate0.words.device:
            raise ValueError("projection weights must be on the input device")
        if tuple(weight.shape) != shape or not weight.is_contiguous():
            raise ValueError(f"shared expert {name} shape must be {shape}")
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != gate0.k or not x.is_contiguous():
        raise ValueError("input must have shape (positive rows, gate.K)")
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.device != gate0.words.device:
        raise ValueError("input must be CUDA bf16")
    if ids.ndim != 2 or ids.shape[0] != x.shape[0] or ids.shape[1] <= 0 or ids.dtype != torch.int64:
        raise ValueError("ids must be int64 with shape (rows, slots)")
    if ids.device != x.device or not torch.isfinite(ids.float()).all():
        raise ValueError("ids must be finite and on the input device")
    if int(ids.min()) < 0 or int(ids.max()) >= len(experts):
        raise ValueError("expert id is out of range; the shared slot is not an expert id")
    if weights.shape != ids.shape or weights.dtype != torch.float32 or weights.device != x.device:
        raise ValueError("weights must be fp32 with the same shape as ids")
    if not torch.isfinite(weights).all():
        raise ValueError("routed weights must be finite")
    if out.shape != (x.shape[0], down0.n):
        raise ValueError("destination must have shape (rows, down.n)")
    if out.dtype != torch.float32 or out.device != x.device:
        raise ValueError("destination must be fp32 on the input device")
    if not out.is_contiguous():
        raise ValueError("destination must be contiguous and have no internal overlap")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("SwiGLU limit must be finite and positive")
    sources: list[torch.Tensor] = [x, ids, weights, gate_w, up_w, down_w]
    for gate, up, down in experts:
        sources.extend((gate.words, gate.bs, up.words, up.bs, down.words, down.bs))
    if any(out.untyped_storage().data_ptr() == source.untyped_storage().data_ptr() for source in sources):
        raise ValueError("destination must not overlap input or projection storage")
