"""One GLM MoE layer in a single-owner pointer table.

This packs checkpoint tensors the caller already holds, without reading shards
or calling a grouped expert kernel. Expert downs stay distinct owners.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Mapping

import torch

from tensorfold.cuda.nvfp4.linear import Fp4Linear, _round_up_bf16

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ROUTED_KEYS = ("weight", "weight_scale", "weight_scale_2", "input_scale")
PREFIX = "model.language_model."


def _f32_bits(number: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(number)))[0]


def _scalar(name: str, key: str, value: torch.Tensor) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise ValueError(f"{name}: {key} must be a scalar, not {shape}")
    number = float(value.detach().float().reshape(-1)[0])
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{name}: {key} {number!r} is not finite")
    return number


def rounded_alpha(act: float, scale: float) -> float:
    """``act * weight_scale_2``, rounded up to bf16. The stored scale is not inverted."""

    product = torch.tensor(act, dtype=torch.float32) * torch.tensor(scale, dtype=torch.float32)
    return float(_round_up_bf16(product).float().reshape(-1)[0])


def _layer_index(name: str) -> int | None:
    marker = ".layers."
    at = name.find(marker)
    if at < 0:
        return None
    num = name[at + len(marker):].split(".", 1)[0]
    return int(num) if num.isdigit() else None


def storage_arm(name: str) -> str:
    """Which arm a checkpoint name uses. Layer 45 and a draft head are refused."""

    if "draft_head" in name or ".mtp." in name:
        raise ValueError("no draft head on this quant")
    index = _layer_index(name)
    if index == 45:
        raise ValueError("layer 45 is not loaded")
    if name.startswith("model.visual") or name.startswith("visual."):
        return "skip"
    if ".experts." in name:
        return "nvfp4"
    if "shared_experts" in name:
        return "bf16-weight"
    return "bf16"


def prefetch_names(layer: int, experts: int) -> list[str]:
    """Routed NVFP4 keys, plus shared-expert ``.weight`` only. No layer 45."""

    if layer == 45:
        raise ValueError("layer 45 is not loaded")
    if experts < 1:
        raise ValueError("routed layer needs at least one expert")
    prefix = f"{PREFIX}layers.{layer}.mlp."
    names = [
        prefix + f"experts.{e}.{proj}.{key}"
        for e in range(experts) for proj in PROJECTIONS for key in ROUTED_KEYS
    ]
    names += [prefix + f"shared_experts.{proj}.weight" for proj in PROJECTIONS]
    return names


def empty_address_table(experts: int = 288, projections: int = 3, device: str = "cpu") -> torch.Tensor:
    """Pointer slots only. This does not allocate production expert weights."""

    if experts < 1 or projections < 1:
        raise ValueError("address table needs at least one expert and one projection")
    return torch.zeros((experts, projections), dtype=torch.int64, device=device)


def bind_addresses(table: torch.Tensor, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Record ``data_ptr`` into ``table``. The table does not copy tensor storage."""

    if table.dtype != torch.int64 or table.numel() != len(tensors):
        raise ValueError(f"address table {tuple(table.shape)} cannot bind {len(tensors)} tensors")
    flat = table.view(-1)
    for slot, tensor in enumerate(tensors):
        flat[slot] = tensor.data_ptr()
    return table


@dataclass
class RoutedTable:
    """Eager ``Fp4Linear`` owners plus a pointer table into those same buffers."""

    linears: list[Fp4Linear]
    words_ptr: torch.Tensor
    bs_ptr: torch.Tensor
    n: torch.Tensor
    k: torch.Tensor
    alpha: torch.Tensor
    shared: dict[str, torch.Tensor]
    combine_weight: float = 1.0
    experts: int = 0
    layer: int = 0


def owned_nbytes(root: object) -> int:
    """Unique buffer bytes. A shared ``Staging`` or pointer slot is counted once."""

    from tensorfold.cuda.nvfp4.linear import Staging

    total = 0
    seen: set[int] = set()

    def add(value: object) -> None:
        nonlocal total
        if isinstance(value, torch.Tensor):
            if value.data_ptr() not in seen:
                seen.add(value.data_ptr())
                total += value.numel() * value.element_size()
            return
        elif isinstance(value, (Fp4Linear, Staging, RoutedTable)):
            for child in vars(value).values():
                add(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                add(child)
        elif isinstance(value, dict):
            for child in value.values():
                add(child)

    add(root)
    return total


def _ordered(experts: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]]) -> list:
    if not experts:
        raise ValueError("routed layer needs at least one expert")
    keys = sorted(experts)
    if keys != list(range(len(experts))):
        raise ValueError(f"expert ids must be 0..{len(experts) - 1}, not {keys}")
    return [experts[i] for i in keys]


def _scales(blocks: list, proj: str) -> list[tuple[float, float]]:
    scales = []
    for index, block in enumerate(blocks):
        name = f"experts.{index}.{proj}"
        if proj not in block:
            raise ValueError(f"{name}: missing projection")
        tensors = block[proj]
        for key in ROUTED_KEYS:
            if key not in tensors:
                raise ValueError(f"{name}: missing {key}")
        scales.append((_scalar(name, "weight_scale_2", tensors["weight_scale_2"]),
                       _scalar(name, "input_scale", tensors["input_scale"])))
    return scales


def _check_shared(shared: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for proj in PROJECTIONS:
        if proj not in shared or not isinstance(shared[proj], torch.Tensor):
            raise ValueError(f"shared expert {proj} needs a BF16 .weight")
        weight = shared[proj]
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"shared expert {proj} is BF16, not {weight.dtype}")
        out[proj] = weight
    return out


def _refuse_production(blocks: list) -> None:
    if len(blocks) < 32:
        return
    weight = blocks[0]["gate_proj"]["weight"]
    if weight.ndim == 2 and weight.shape[0] >= 2048 and weight.shape[1] >= 1024:
        raise ValueError("synthetic loader refuses production expert weights")


def load_synthetic_layer(
    *,
    layer: int,
    experts: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    shared: Mapping[str, torch.Tensor],
    shards: list[Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]]] | None = None,
) -> RoutedTable:
    """Pack one synthetic layer. Same-expert shard disagreement raises before packing."""

    return _load_layer(layer=layer, experts=experts, shared=shared, shards=shards, synthetic=True)


def load_checkpoint_layer(
    *,
    layer: int,
    experts: Mapping[int, Mapping[str, Mapping[str, torch.Tensor]]],
    shared: Mapping[str, torch.Tensor],
) -> RoutedTable:
    """Pack rank-local checkpoint tensors into the same owners used by eager execution."""

    return _load_layer(layer=layer, experts=experts, shared=shared, shards=None, synthetic=False)


def _load_layer(*, layer: int, experts: Mapping, shared: Mapping, shards: list | None,
                synthetic: bool) -> RoutedTable:
    if layer == 45:
        raise ValueError("layer 45 is not loaded")
    if layer < 0:
        raise ValueError(f"layer {layer} is not a routed layer")
    blocks = _ordered(experts)
    scales = {proj: _scales(blocks, proj) for proj in PROJECTIONS}
    for extra in shards or ():
        _same_shard(blocks, _ordered(extra))
    shared_weights = _check_shared(shared)
    if synthetic:
        _refuse_production(blocks)
    built: list[Fp4Linear] = []
    try:
        for index, block in enumerate(blocks):
            for proj in PROJECTIONS:
                tensors = block[proj]
                weight_scale_2, act = scales[proj][index]
                built.append(Fp4Linear.from_checkpoint(
                    tensors["weight"], tensors["weight_scale"], weight_scale_2, act=act))
        return _assemble(layer, built, scales, shared_weights)
    except Exception:
        built.clear()
        raise


def _same_shard(primary: list, extra: list) -> None:
    if len(primary) != len(extra):
        raise ValueError("shard expert count disagrees")
    for proj in PROJECTIONS:
        left = _scales(primary, proj)
        right = _scales(extra, proj)
        for index, (a, b) in enumerate(zip(left, right)):
            for key, x, y in zip(("weight_scale_2", "input_scale"), a, b):
                if _f32_bits(x) != _f32_bits(y):
                    raise ValueError(f"experts.{index}.{proj} {key} disagrees across shards")


def _assemble(layer: int, built: list[Fp4Linear], scales: dict[str, list[tuple[float, float]]],
              shared: dict[str, torch.Tensor]) -> RoutedTable:
    experts = len(built) // len(PROJECTIONS)
    device = built[0].words.device
    words_ptr = torch.empty((experts, len(PROJECTIONS)), dtype=torch.int64, device=device)
    bs_ptr = torch.empty_like(words_ptr)
    n = torch.empty((experts, len(PROJECTIONS)), dtype=torch.int32, device=device)
    k = torch.empty_like(n)
    alpha = torch.empty((experts, len(PROJECTIONS)), dtype=torch.float32, device=device)
    for slot, linear in enumerate(built):
        row, col = divmod(slot, len(PROJECTIONS))
        words_ptr[row, col] = linear.words.data_ptr()
        bs_ptr[row, col] = linear.bs.data_ptr()
        n[row, col] = linear.n
        k[row, col] = linear.k
        weight_scale_2, act = scales[PROJECTIONS[col]][row]
        alpha[row, col] = rounded_alpha(act, weight_scale_2)
    return RoutedTable(built, words_ptr, bs_ptr, n, k, alpha, shared, 1.0, experts, layer)
