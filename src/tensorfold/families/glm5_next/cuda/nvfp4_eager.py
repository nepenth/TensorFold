"""Dense NVFP4 eager oracle for the GLM5-next MLP.

Gate and up share input quantization only for matching activation factors.
The activation is the CUDA glue SwiGLU and down writes the caller's fp32 buffer.
"""

from __future__ import annotations

import math

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

    gate_up = checkpoint.matmul_group(x, [gate, up])
    if gate_up is None:
        raise ValueError("gate/up cannot share input quantization")
    gu = torch.cat(gate_up, dim=1)
    activated = torch.empty((x.shape[0], gate.n), dtype=torch.bfloat16, device=x.device)
    sums = torch.empty((x.shape[0], gate.n // 64), dtype=torch.float32, device=x.device)
    glue.swiglu(gu, activated, sums, limit)
    return checkpoint.matmul(checkpoint.A4, activated, down, out=out, f32=True)
