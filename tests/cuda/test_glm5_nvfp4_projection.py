"""One dense GLM NVFP4 projection on the existing lane and prompt kernels. No checkpoint pull."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("GLM NVFP4 projection kernels need an SM 12 GPU", allow_module_level=True)

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.families.glm5_next.cuda.nvfp4_proj import dense_projection

# Rank-local dense gate: packed [6144, 2048] is logical N=6144, K=4096.
DENSE_N, DENSE_K = 6144, 4096
RECORDED_SPLIT_K = 2


def _tensors(n: int, k: int, weight_scale_2: float, input_scale: float) -> dict:
    # e4m3 bytes 0x7f and 0xff are NaN. A random scale is not a finite projection.
    return {
        "weight": torch.full((n, k // 2), 0x11, dtype=torch.uint8, device="cuda"),
        "weight_scale": torch.full((n, k // 16), 0x38, dtype=torch.uint8, device="cuda"),
        "weight_scale_2": torch.tensor(weight_scale_2, device="cuda"),
        "input_scale": torch.tensor(input_scale, device="cuda"),
    }


def test_gate_and_up_keep_distinct_weight_scales():
    gate = dense_projection("model.layers.0.mlp.gate_proj", _tensors(64, 64, 1.25, 0.5))
    up = dense_projection("model.layers.0.mlp.up_proj", _tensors(64, 64, 2.5, 0.5))
    assert gate is not None and up is not None
    assert gate.act == pytest.approx(0.5)
    assert up.act == pytest.approx(0.5)
    assert gate.scale == pytest.approx(1.25)
    assert up.scale == pytest.approx(2.5)
    assert gate.scale != up.scale


def test_one_dense_gate_runs_on_the_lane_and_the_prompt_gemm():
    lin = dense_projection("model.layers.0.mlp.gate_proj", _tensors(DENSE_N, DENSE_K, 1.25, 0.5))
    assert lin is not None
    assert lin.n == DENSE_N and lin.k == DENSE_K
    assert lin.scale == pytest.approx(1.25)
    assert lin.act == pytest.approx(0.5)
    row = torch.randn(1, DENSE_K, dtype=torch.bfloat16, device="cuda")
    lane = checkpoint.matmul(checkpoint.A4, row, lin)
    prompt = checkpoint.prompt(checkpoint.A4, row, lin)
    assert lane.shape == (1, DENSE_N) and prompt.shape == (1, DENSE_N)
    assert lane.dtype == torch.bfloat16 and prompt.dtype == torch.bfloat16
    assert torch.isfinite(lane).all() and torch.isfinite(prompt).all()
    # Recorded for this shape. Lane and prompt are different reductions, so they are not compared.
    assert qmm.split_k(lin.n, lin.k) == RECORDED_SPLIT_K
