"""Task 10C: eager lane prefill with a stable pack and slot-ordered combine.

The pack runs on the routing device. Numeric work reuses the lane eager oracle,
including its BF16 shared branch; this is deliberately an eager, synchronizing
path. It does not qualify prefill for CUDA graphs or use the prompt backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorfold.families.glm5_next.cuda.nvfp4_table import RoutedTable


def _tile_capacity(assignments: int, experts: int, tile_size: int) -> int:
    # Each active expert needs one assignment for its first tile and tile_size
    # more for each additional tile. This bounds EVERY distribution, not just
    # the balanced one. Empty experts consume no descriptor.
    active = min(assignments, experts)
    return active + (assignments - active) // tile_size


@dataclass(frozen=True)
class PrefillPack:
    """Active indices are views; a workspace reuse invalidates previous packs.

    Offsets include the terminal exclusive prefix sum. Descriptor triples are
    (expert, packed_start, packed_end), with an exclusive end and no padded
    assignment in the range. Unused descriptor rows contain -1.
    """

    counts: torch.Tensor
    offsets: torch.Tensor
    assignment_ids: torch.Tensor
    token_ids: torch.Tensor
    slot_ids: torch.Tensor
    tile_descriptors: torch.Tensor
    num_tiles: int


@dataclass(frozen=True)
class PrefillWorkspace:
    """Reusable route metadata and fp32 slots for at most max_tokens rows."""

    max_tokens: int
    slots: int
    hidden: int
    num_experts: int
    tile_size: int
    counts: torch.Tensor
    offsets: torch.Tensor
    assignment_ids: torch.Tensor
    token_ids: torch.Tensor
    slot_ids: torch.Tensor
    tile_descriptors: torch.Tensor
    assignment_outputs: torch.Tensor
    combine_weights: torch.Tensor

    @classmethod
    def allocate(cls, max_tokens: int, slots: int, hidden: int, num_experts: int,
                 *, device: torch.device | str, tile_size: int = 16) -> PrefillWorkspace:
        if max_tokens < 0 or min(slots, hidden, num_experts, tile_size) <= 0:
            raise ValueError("capacity must be nonnegative and slots/hidden/experts/tile_size positive")
        assignments = max_tokens * slots

        def integers(*shape):
            return torch.full(shape, -1, dtype=torch.int64, device=device)

        return cls(
            max_tokens, slots, hidden, num_experts, tile_size,
            torch.zeros(num_experts, dtype=torch.int64, device=device),
            torch.zeros(num_experts + 1, dtype=torch.int64, device=device),
            integers(assignments), integers(assignments), integers(assignments),
            integers(_tile_capacity(assignments, num_experts, tile_size), 3),
            torch.empty((max_tokens, slots + 1, hidden), dtype=torch.float32, device=device),
            torch.empty((max_tokens, slots + 1), dtype=torch.float32, device=device),
        )

    def _buffers(self):
        return (self.counts, self.offsets, self.assignment_ids, self.token_ids,
                self.slot_ids, self.tile_descriptors, self.assignment_outputs, self.combine_weights)


def pack_assignments(routes: torch.Tensor, num_experts: int, tile_size: int = 16,
                     *, workspace: PrefillWorkspace | None = None) -> PrefillPack:
    """Stably group token-major assignments by expert, including zero weights.

    CUDA and CPU int64 routing are supported. Reuse clears ALL descriptor and
    index padding so a smaller/empty batch cannot replay a previous assignment.
    """
    if routes.ndim != 2 or routes.dtype != torch.int64 or routes.shape[1] <= 0:
        raise ValueError("routes must be int64 with shape [token, positive slots]")
    if num_experts <= 0 or tile_size <= 0:
        raise ValueError("num_experts and tile_size must be positive")
    flat = routes.reshape(-1)
    if flat.numel() and bool(((flat < 0) | (flat >= num_experts)).any()):
        raise ValueError("route id out of range")
    rows, slots = routes.shape
    if workspace is not None:
        _check_workspace(workspace, rows, slots, num_experts, tile_size, routes.device)
        if any(_overlap(routes, buffer) for buffer in workspace._buffers()):
            raise ValueError("routing must not overlap workspace storage")
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    order = torch.argsort(flat, stable=True)
    tokens, slot_ids = order // slots, order % slots
    # Eager descriptor construction synchronizes only once for the counts.
    descriptors = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        descriptors.extend((expert, tile_start, min(tile_start + tile_size, end))
                           for tile_start in range(start, end, tile_size))
        start = end
    if workspace is None:
        tiles = counts.new_full((_tile_capacity(flat.numel(), num_experts, tile_size), 3), -1)
    else:
        workspace.counts.copy_(counts)
        workspace.offsets.copy_(offsets)
        counts, offsets = workspace.counts, workspace.offsets
        for buffer, values in ((workspace.assignment_ids, order), (workspace.token_ids, tokens),
                               (workspace.slot_ids, slot_ids)):
            buffer.fill_(-1)
            buffer[:flat.numel()].copy_(values)
        order = workspace.assignment_ids[:flat.numel()]
        tokens, slot_ids = workspace.token_ids[:flat.numel()], workspace.slot_ids[:flat.numel()]
        tiles = workspace.tile_descriptors
        tiles.fill_(-1)
    if descriptors:
        tiles[:len(descriptors)].copy_(counts.new_tensor(descriptors))
    return PrefillPack(counts, offsets, order, tokens, slot_ids, tiles, len(descriptors))


def _check_workspace(workspace, rows, slots, experts, tile_size, device):
    if not isinstance(workspace, PrefillWorkspace):
        raise TypeError("expected PrefillWorkspace")
    if (rows > workspace.max_tokens or slots != workspace.slots or experts != workspace.num_experts or
            tile_size != workspace.tile_size or any(t.device != device for t in workspace._buffers())):
        raise ValueError("workspace capacity, slots, experts, tile_size, and device must match")


def _overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    a, b = left.untyped_storage(), right.untyped_storage()
    return bool(a.nbytes() and b.nbytes() and a.data_ptr() == b.data_ptr())


def routed_prefill(x: torch.Tensor, table: RoutedTable, ids: torch.Tensor, weights: torch.Tensor,
                   out: torch.Tensor, *, workspace: PrefillWorkspace | None = None,
                   tile_size: int = 16, limit: float = 10.0) -> torch.Tensor:
    """Write a multi-token routed MLP into caller-owned contiguous fp32 out.

    This slice uses the table's eager Fp4Linear owners and their activation and
    weight factors. Each selected assignment (even weight zero) is evaluated
    exactly once by the lane oracle. Expert order controls execution only;
    results are scattered to [token, slot], shared is appended at weight one,
    and glue.combine reduces in the ORIGINAL slot order. There is no fp32
    atomic accumulation. Only logical rows are written, including on reuse.
    """
    from tensorfold.families.glm5_next.cuda import glue, nvfp4_eager
    from tensorfold.families.glm5_next.cuda.nvfp4_table import RoutedTable

    if not isinstance(table, RoutedTable):
        raise TypeError("expected RoutedTable")
    experts_count = table.words_ptr.shape[0]
    if experts_count <= 0 or len(table.linears) != 3 * experts_count:
        raise ValueError("table needs one gate/up/down owner triple per expert")
    experts = tuple(tuple(table.linears[3 * e:3 * e + 3]) for e in range(experts_count))
    hidden = experts[0][2].n
    block = min(1024, hidden)
    if hidden <= 0 or hidden % block or block & (block - 1):
        raise ValueError("hidden must fit whole glue.combine blocks")
    if tile_size <= 0 or not math.isfinite(limit) or limit <= 0:
        raise ValueError("tile_size and finite SwiGLU limit must be positive")
    if (x.ndim != 2 or x.shape[1] != experts[0][0].k or not x.is_cuda or
            x.dtype != torch.bfloat16 or not x.is_contiguous()):
        raise ValueError("input must be contiguous CUDA bf16 with shape (rows, gate.K)")
    if ids.ndim != 2 or ids.shape[0] != x.shape[0] or ids.shape[1] <= 0 or ids.dtype != torch.int64:
        raise ValueError("ids must be int64 with shape (rows, positive slots)")
    if ids.device != x.device or weights.device != x.device or weights.dtype != torch.float32:
        raise ValueError("routing must be on the input device with fp32 weights")
    if weights.shape != ids.shape or not bool(torch.isfinite(weights).all()):
        raise ValueError("weights must be finite with the same shape as ids")
    if out.shape != (x.shape[0], hidden) or out.dtype != torch.float32 or out.device != x.device:
        raise ValueError("destination must be fp32 on the input device with shape (rows, hidden)")
    if not out.is_contiguous():
        raise ValueError("destination must be contiguous and have no internal overlap")
    # The existing oracle validates every owner and shared operand before work.
    # Its positive-row restriction is bypassed only for a no-work empty batch.
    if x.shape[0]:
        nvfp4_eager._refuse_routed(x, experts, table.shared, ids, weights, out, "lane", limit)
    sources = [x, ids, weights, *table.shared.values(), table.words_ptr, table.bs_ptr,
               table.n, table.k, table.alpha]
    sources.extend(t for triple in experts for lin in triple for t in (lin.words, lin.bs))
    if any(_overlap(out, source) for source in sources):
        raise ValueError("destination must not overlap input, routing, table, shared, or projection storage")
    rows, slots = ids.shape
    if workspace is None:
        workspace = PrefillWorkspace.allocate(rows, slots, hidden, experts_count,
                                               device=x.device, tile_size=tile_size)
    _check_workspace(workspace, rows, slots, experts_count, tile_size, x.device)
    if workspace.hidden != hidden:
        raise ValueError("workspace hidden width must match the destination")
    if any(_overlap(buffer, source) for buffer in workspace._buffers() for source in [out, *sources]):
        raise ValueError("workspace must not overlap input, destination, routing, table, or weights")
    packed = pack_assignments(ids, experts_count, tile_size, workspace=workspace)
    if not rows:
        return out
    ey = workspace.assignment_outputs[:rows]
    combined = workspace.combine_weights[:rows]
    token_ids, slot_ids = packed.token_ids.tolist(), packed.slot_ids.tolist()
    for expert, start, end in packed.tile_descriptors[:packed.num_tiles].tolist():
        gate, up, down = experts[expert]
        for index in range(start, end):
            token, slot = token_ids[index], slot_ids[index]
            ey[token, slot] = nvfp4_eager._assignment(x[token:token + 1], gate, up, down, "lane", limit)
    ey[:, slots] = nvfp4_eager._shared_mlp(x, table.shared, limit)
    combined[:, :slots] = weights
    combined[:, slots] = 1.0
    glue.combine(ey, combined, out)
    return out
