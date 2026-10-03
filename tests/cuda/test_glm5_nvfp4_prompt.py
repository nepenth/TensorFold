"""Prompt eager oracle. Lane and prompt are not a bitwise pair."""

# ruff: noqa: E402 -- CUDA availability MUST be checked before backend imports.

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("GLM NVFP4 eager oracle needs an SM 12 GPU", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear
from tensorfold.families.glm5_next.cuda import glue
from tensorfold.families.glm5_next.cuda.nvfp4_eager import dense_prompt


@pytest.fixture(scope="module")
def projections():
    lins = []
    for seed, factor, act in ((81, 0.03125, 0.25), (82, 0.0625, 0.25), (83, 0.015625, 0.125)):
        rng = torch.Generator().manual_seed(seed)
        weight = torch.randint(0, 16, (128, 64), generator=rng, dtype=torch.uint8).cuda()
        scales = torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda")
        lins.append(Fp4Linear.from_checkpoint(weight, scales, factor, act=act))
    return tuple(lins)


def _input(rows=3):
    rng = torch.Generator().manual_seed(91)
    return torch.randn((rows, 128), generator=rng).to(torch.bfloat16).cuda()


def _out(rows=3):
    return torch.full((rows, 128), float("nan"), dtype=torch.float32, device="cuda")


def _bits(x):
    assert x.dtype == torch.float32
    return x.view(torch.int32)


def _forbid_work(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("refusal must precede quantization, multiply, and SwiGLU")

    for name in ("quant4", "matmul_group", "matmul", "prompt", "_lane", "_gemm", "mlp_prompt"):
        monkeypatch.setattr(checkpoint, name, fail)
    monkeypatch.setattr(glue, "swiglu", fail)


@pytest.mark.parametrize("rows", [1, 3])
def test_prompt_repeat_and_two_prompt_calls_are_bitwise(projections, rows):
    gate, up, down = projections
    x = _input(rows)
    first, second = _out(rows), _out(rows)
    assert dense_prompt(x, gate, up, down, first) is first
    assert dense_prompt(x, gate, up, down, second) is second
    assert torch.isfinite(first).all() and first.abs().max() > 0
    assert torch.equal(_bits(first), _bits(second))

    gu = torch.cat((checkpoint.prompt(checkpoint.A4, x, gate), checkpoint.prompt(checkpoint.A4, x, up)), dim=1)
    activation = torch.empty((rows, 128), dtype=torch.bfloat16, device="cuda")
    sums = torch.empty((rows, 2), dtype=torch.float32, device="cuda")
    glue.swiglu(gu, activation, sums, 10.0)
    reference = checkpoint.prompt(checkpoint.A4, activation, down, f32=True)
    assert torch.equal(_bits(first), _bits(reference))


def test_prompt_uses_grouped_prompt_then_prompt_down(projections, monkeypatch):
    gate, up, down = projections
    x, out = _input(), _out()
    calls = []
    grouped, prompt = checkpoint.matmul_group, checkpoint.prompt

    def record_group(values, lins, prompt_rows=False, outs=None, tile=0):
        calls.append(("group", prompt_rows, lins))
        return grouped(values, lins, prompt_rows, outs, tile)

    def record_prompt(mode, values, lin, out=None, f32=False, tile=0):
        calls.append(("prompt", mode, lin, out, f32))
        return prompt(mode, values, lin, out, f32, tile)

    def fail(*args, **kwargs):
        pytest.fail("prompt oracle must not use the lane kernel or mlp_prompt")

    monkeypatch.setattr(checkpoint, "matmul_group", record_group)
    monkeypatch.setattr(checkpoint, "prompt", record_prompt)
    monkeypatch.setattr(checkpoint, "_lane", fail)
    monkeypatch.setattr(checkpoint, "matmul", fail)
    monkeypatch.setattr(checkpoint, "mlp_prompt", fail)
    assert dense_prompt(x, gate, up, down, out) is out
    assert calls[0][0] == "group" and calls[0][1] is True and calls[0][2] == [gate, up]
    assert calls[1][0] == "prompt" and calls[1][2] is down and calls[1][3] is out and calls[1][4] is True


def test_activation_factor_mismatch_raises_before_work(projections, monkeypatch):
    gate, up, down = projections
    up = replace(up, act=gate.act + 1e-12)
    x, out = _input(), _out()
    before = _bits(out).clone()
    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="activation factors must match"):
        dense_prompt(x, gate, up, down, out)
    assert torch.equal(_bits(out), before)


def test_gate_above_ten_differs_numerically_from_mlp_prompt():
    def linear(factor):
        return Fp4Linear.from_checkpoint(
            torch.full((128, 64), 0x22, dtype=torch.uint8, device="cuda"),
            torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda"),
            factor, act=0.5)

    gate, up, down = linear(0.25), linear(1 / 128), linear(1 / 128)
    x = torch.ones((3, 128), dtype=torch.bfloat16, device="cuda")
    assert (gate(x) > 10).all()
    eager, repeated = _out(), _out()
    dense_prompt(x, gate, up, down, eager)
    dense_prompt(x, gate, up, down, repeated)
    assert torch.equal(_bits(eager), _bits(repeated))
    unclamped = _out()
    dense_prompt(x, gate, up, down, unclamped, limit=100.0)
    assert (unclamped > eager * 2).all()
    fused = checkpoint.mlp_prompt(x, gate, up, down)
    assert fused is not None and torch.isfinite(fused).all()
    assert (fused.float() > eager * 2).all()


@pytest.mark.parametrize("kind", ["noncontiguous", "multirow", "wrong_dtype", "wrong_shape", "internal_overlap"])
def test_refused_destinations_raise_before_work(projections, monkeypatch, kind):
    x = _input()
    if kind == "noncontiguous":
        backing = torch.full((3, 256), -123.0, dtype=torch.float32, device="cuda")
        out, message = backing[:, ::2], "contiguous"
    elif kind == "multirow":
        backing = torch.full((6, 128), -123.0, dtype=torch.float32, device="cuda")
        out, message = backing[::2], "contiguous"
        assert out.shape == (3, 128) and not out.is_contiguous()
    elif kind == "wrong_dtype":
        backing = torch.full((3, 128), -123.0, dtype=torch.bfloat16, device="cuda")
        out, message = backing, "fp32"
    elif kind == "wrong_shape":
        backing = torch.full((3, 129), -123.0, dtype=torch.float32, device="cuda")
        out, message = backing, "shape"
    else:
        backing = torch.full((1, 128), -123.0, dtype=torch.float32, device="cuda")
        out, message = backing.expand(3, 128), "overlap"
    before = backing.clone()
    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match=message):
        dense_prompt(x, *projections, out)
    assert torch.equal(backing, before)


def test_caller_buffer_canary_past_write(projections):
    x = _input()
    backing = torch.full((16 + 3 * 128 + 64,), -123.0, dtype=torch.float32, device="cuda")
    out = backing[16:16 + 3 * 128].view(3, 128)
    prefix = _bits(backing[:16]).clone()
    canary = _bits(backing[16 + 3 * 128:]).clone()
    pointer = out.data_ptr()
    assert dense_prompt(x, *projections, out) is out
    assert out.data_ptr() == pointer and torch.isfinite(out).all()
    assert torch.equal(_bits(backing[:16]), prefix)
    assert torch.equal(_bits(backing[16 + 3 * 128:]), canary)


def test_destination_overlapping_input_is_refused(projections, monkeypatch):
    backing = torch.ones((3, 128), dtype=torch.float32, device="cuda")
    x = backing.view(torch.bfloat16)[:, 64:192]
    before = _bits(backing).clone()
    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="overlap"):
        dense_prompt(x, *projections, backing)
    assert torch.equal(_bits(backing), before)
