"""One GLM dense MLP projection in the checkpoint's own NVFP4 math.

Routed experts are not this module. ``weights.load`` still refuses a modelopt tree.
The stored ``weight_scale_2`` is passed through. It is not the compressed-tensors reciprocal.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

from tensorfold.cuda.nvfp4.linear import Fp4Linear


def dense_projection(name: str, tensors: Mapping[str, torch.Tensor]) -> Fp4Linear | None:
    """Map one dense projection. Vision is skipped. Routed experts are refused."""

    if name.startswith("model.visual") or name.startswith("visual."):
        return None
    if ".experts." in name:
        raise ValueError("routed experts are not wired")
    for key in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        if key not in tensors:
            raise ValueError(f"{name}: missing {key}")
    return Fp4Linear.from_checkpoint(
        tensors["weight"], tensors["weight_scale"], _scalar(name, "weight_scale_2", tensors["weight_scale_2"]),
        act=_scalar(name, "input_scale", tensors["input_scale"]))


def _scalar(name: str, key: str, value: torch.Tensor) -> float:
    """The stored factor, unchanged. A non-finite or non-scalar value raises before packing."""

    if value.numel() != 1:
        raise ValueError(f"{name}: {key} must be a scalar, not {tuple(value.shape)}")
    number = float(value.detach().float().reshape(-1)[0])
    if not math.isfinite(number):
        raise ValueError(f"{name}: {key} {number!r} is not finite")
    return number
