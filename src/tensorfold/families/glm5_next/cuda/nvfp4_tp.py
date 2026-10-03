"""Task 9A synthetic two-rank arithmetic on one CUDA device.

CPU owners are supplied by the caller; this module performs no checkpoint I/O.
The existing lane projection evaluates the full and rank shapes. Gate/up
concatenate rank 0 then rank 1; down adds their fp32 partials in that order.
TP split to unsplit has a named envelope with no frozen numeric bound.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear
from tensorfold.families.glm5_next.cuda import split
from tensorfold.families.glm5_next.cuda.nvfp4_proj import dense_projection

TP_ENVELOPE = "glm5_nvfp4_tp_split_to_unsplit_lane_fp32"
RANK_ORDER = (0, 1)
_DTYPES = {"weight": "U8", "weight_scale": "F8_E4M3", "weight_scale_2": "F32", "input_scale": "F32"}


@dataclass(frozen=True)
class TPRecord:
    """Shape arithmetic and comparison policy, independent of completion order."""

    kind: str
    full_shape: tuple[int, int]  # logical (N, K)
    rank_shapes: tuple[tuple[int, int], tuple[int, int]]
    full_split_k: int
    rank_split_k: tuple[int, int]
    rank_order: tuple[int, int] = RANK_ORDER
    envelope: str = TP_ENVELOPE
    envelope_frozen: bool = False


@dataclass(frozen=True)
class TPResult:
    """Full output, rank outputs, ordered combination, and an error observation.

    ``max_abs_error`` is an observation, NEVER a numeric envelope pass.
    """

    full: torch.Tensor
    rank_outputs: tuple[torch.Tensor, torch.Tensor]
    combined: torch.Tensor
    max_abs_error: float
    record: TPRecord


@dataclass(frozen=True)
class SyntheticTP:
    """Full and two rank-local synthetic owners on the same device."""

    full: Fp4Linear
    ranks: tuple[Fp4Linear, Fp4Linear]
    record: TPRecord

    def project(self, x: torch.Tensor) -> TPResult:
        """Run the existing A4 lane and combine fp32 outputs in rank order."""
        if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != self.full.k:
            raise ValueError("input must have shape (positive rows, full.K)")
        if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != self.full.words.device:
            raise ValueError("input must be bf16 on the owners' single CUDA device")
        if self.record.kind == "row":
            inputs = (x, x)
        else:
            # Reuse split_bytes for activation columns as well as stored weights.
            host = x.detach().cpu().contiguous()
            inputs = tuple(_part(host, "BF16", "col", rank).to(x.device) for rank in RANK_ORDER)
        full = checkpoint.matmul(checkpoint.A4, x, self.full, f32=True)
        outputs = tuple(checkpoint.matmul(checkpoint.A4, inputs[rank], self.ranks[rank], f32=True)
                        for rank in RANK_ORDER)
        if self.record.kind == "row":
            combined = torch.cat(outputs, dim=1)
        else:
            # No bf16 rounding of partials, atomic adds, or completion-order reduction.
            combined = torch.add(outputs[0], outputs[1])
        error = float((combined - full).abs().max())
        return TPResult(full, outputs, combined, error, self.record)


def synthetic_tp(name: str, tensors: Mapping[str, torch.Tensor], *,
                 device: torch.device | str = "cuda") -> SyntheticTP:
    """Pack caller-created CPU tensors into full and TP=2 projection owners.

    ``name`` is a dense gate/up/down projection prefix. All shapes and factors,
    including both column-split rank widths, MUST be legal before any CUDA work.
    Scalars replicate through ``split.rule`` and ``split.split_bytes`` too.
    """
    _refuse(name, tensors)
    kind = split.rule(f"{name}.weight")
    # Split every source before even the full owner's packing kernel can launch.
    parts = tuple({key: _part(tensors[key], dtype, split.rule(f"{name}.{key}"), rank)
                   for key, dtype in _DTYPES.items()} for rank in RANK_ORDER)
    n, packed_k = tensors["weight"].shape
    full_shape = (n, 2 * packed_k)
    rank_shapes = tuple((part["weight"].shape[0], 2 * part["weight"].shape[1]) for part in parts)
    from tensorfold.cuda.kernels import qmm

    record = TPRecord(kind, full_shape, rank_shapes, qmm.split_k(*full_shape),
                      tuple(qmm.split_k(*shape) for shape in rank_shapes))
    full = dense_projection(name, {key: tensors[key].to(device) for key in _DTYPES})
    ranks = tuple(dense_projection(name, {key: part[key].to(device) for key in _DTYPES}) for part in parts)
    return SyntheticTP(full, ranks, record)


def _part(tensor: torch.Tensor, dtype: str, kind: str, rank: int) -> torch.Tensor:
    """Preserve stored bytes using the one existing splitter, with independent ownership."""
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    data, shape = split.split_bytes(raw, list(tensor.shape), tensor.element_size(), kind, rank, dtype=dtype)
    return torch.from_numpy(data.copy()).view(tensor.dtype).reshape(shape)


def _refuse(name: str, tensors: Mapping[str, torch.Tensor]) -> None:
    if ".experts." in name or name.rsplit(".", 1)[-1] not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError("synthetic TP needs a dense gate/up/down projection name")
    kind = split.rule(f"{name}.weight")
    for key in _DTYPES:
        if key not in tensors:
            raise ValueError(f"{name}: missing {key}")
        if tensors[key].device.type != "cpu":
            raise ValueError("synthetic source owners must be CPU tensors")
    weight, scales = tensors["weight"], tensors["weight_scale"]
    if weight.ndim != 2 or weight.dtype != torch.uint8 or min(weight.shape) <= 0:
        raise ValueError("weight must be positive 2-D packed U8")
    n, packed_k = weight.shape
    if kind == "col":
        # In particular packed width 96 MUST fail here, before any packing or quantization.
        split.nvfp4_column_legal(list(weight.shape), "U8")
        split.nvfp4_column_legal(list(scales.shape), "F8_E4M3")
    elif n % 2:
        raise ValueError("row split needs an even N")
    if (2 * packed_k) % 64:
        raise ValueError("logical K must be a multiple of 64")
    if scales.shape != (n, packed_k // 8) or scales.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("weight_scale must be E4M3 bytes with shape (N, K/16)")
    if not torch.isfinite(scales.contiguous().view(torch.float8_e4m3fn).float()).all():
        raise ValueError("weight_scale must be finite")
    for key in ("weight_scale_2", "input_scale"):
        value = tensors[key]
        if value.numel() != 1 or value.dtype != torch.float32 or not math.isfinite(float(value)):
            raise ValueError(f"{key} must be a finite fp32 scalar")
        if key == "input_scale" and float(value) <= 0:
            raise ValueError("input_scale must be positive")
