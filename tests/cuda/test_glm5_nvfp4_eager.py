"""Dense lane oracle contracts using synthetic weights; no checkpoint pull."""

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
from tensorfold.families.glm5_next.cuda.nvfp4_eager import dense_eager


@pytest.fixture(scope="module")
def projections():
    lins = []
    # Different weight factors MUST survive common-input quantization.
    for seed, factor, act in ((81, 0.03125, 0.25), (82, 0.0625, 0.25), (83, 0.015625, 0.125)):
        rng = torch.Generator().manual_seed(seed)
        weight = torch.randint(0, 256, (128, 64), generator=rng, dtype=torch.uint8).cuda()
        scales = torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda")  # e4m3 1.0
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

    for name in ("quant4", "matmul_group", "matmul", "_lane"):
        monkeypatch.setattr(checkpoint, name, fail)
    monkeypatch.setattr(glue, "swiglu", fail)


@pytest.mark.parametrize("rows", [1, 3, 17, 33])
def test_lane_eager_repeat_and_explicit_lane_composition_are_bitwise(projections, rows):
    gate, up, down = projections
    x = _input(rows)
    first, second = _out(rows), _out(rows)
    assert dense_eager(x, gate, up, down, first) is first
    assert dense_eager(x, gate, up, down, second) is second
    assert torch.isfinite(first).all()
    assert first.abs().max() > 0
    assert torch.equal(_bits(first), _bits(second))

    # Independent quantizations also give the same lane bits for matching factors.
    gu = torch.cat((gate(x), up(x)), dim=1)
    activation = torch.empty_like(x)
    sums = torch.empty((rows, 2), dtype=torch.float32, device="cuda")
    glue.swiglu(gu, activation, sums, 10.0)
    reference = checkpoint.matmul(checkpoint.A4, activation, down, f32=True)
    assert torch.equal(_bits(first), _bits(reference))


def test_common_quantization_then_glue_then_one_fp32_down(projections, monkeypatch):
    gate, up, down = projections
    x, out = _input(), _out()
    events, quantizations, lanes, activations = [], [], [], []
    quant4, lane, swiglu = checkpoint.quant4, checkpoint._lane, glue.swiglu

    def record_quant(values, act, tb=0):
        events.append("quant")
        quantizations.append((values, act, tb))
        return quant4(values, act, tb)

    def record_lane(mode, rows, lin, y, f32):
        events.append("lane")
        lanes.append((mode, rows, lin, y, f32))
        return lane(mode, rows, lin, y, f32)

    def record_swiglu(gu, activated, sums, limit):
        events.append("swiglu")
        assert gu.shape == (3, 256) and gu.is_contiguous()
        assert activated.dtype == torch.bfloat16
        assert sums.shape == (3, 2) and sums.dtype == torch.float32
        assert limit == 10.0
        activations.append(activated)
        return swiglu(gu, activated, sums, limit)

    monkeypatch.setattr(checkpoint, "quant4", record_quant)
    monkeypatch.setattr(checkpoint, "_lane", record_lane)
    monkeypatch.setattr(glue, "swiglu", record_swiglu)
    assert dense_eager(x, gate, up, down, out) is out
    assert events == ["quant", "lane", "lane", "swiglu", "quant", "lane"]
    assert len(quantizations) == 2
    assert quantizations[0][0] is x and quantizations[0][1:] == (gate.act, 0)
    assert quantizations[1][0] is activations[0] and quantizations[1][1:] == (down.act, 0)
    assert lanes[0][2] is gate and lanes[1][2] is up and lanes[2][2] is down
    assert lanes[0][1] is lanes[1][1]
    assert all(entry[0] == checkpoint.A4 for entry in lanes)
    assert [entry[4] for entry in lanes] == [False, False, True]
    assert lanes[2][3] is out


def test_activation_factor_mismatch_raises_before_work(projections, monkeypatch):
    gate, up, down = projections
    # An adjacent float factor MUST also be refused; no approximate equality.
    up = replace(up, act=gate.act + 1e-12)
    x, out = _input(), _out()
    before = _bits(out).clone()
    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="activation factors must match"):
        dense_eager(x, gate, up, down, out)
    assert torch.equal(_bits(out), before)


def test_gate_above_ten_is_clamped_and_differs_from_unclamped_prompt():
    # Constant positive weights make clipping dominate any reduction differences.
    def linear(factor):
        return Fp4Linear.from_checkpoint(
            torch.full((128, 64), 0x22, dtype=torch.uint8, device="cuda"),  # e2m1 1.0
            torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda"),
            factor, act=0.5)

    gate, up, down = linear(0.25), linear(1 / 128), linear(1 / 128)
    x = torch.ones((3, 128), dtype=torch.bfloat16, device="cuda")
    assert (gate(x) > 10).all()
    eager, repeated = _out(), _out()
    dense_eager(x, gate, up, down, eager)
    dense_eager(x, gate, up, down, repeated)
    assert torch.equal(_bits(eager), _bits(repeated))
    unclamped_lane = _out()
    dense_eager(x, gate, up, down, unclamped_lane, limit=100.0)
    assert (unclamped_lane > eager * 2).all()
    prompt = checkpoint.mlp_prompt(x, gate, up, down)
    assert prompt is not None and torch.isfinite(prompt).all()
    # Lane and prompt use different reductions: compare the clipping effect numerically.
    assert (prompt.float() > eager * 2).all()


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
        dense_eager(x, *projections, out)
    assert torch.equal(backing, before)


@pytest.mark.parametrize("width", [127, 128])
def test_caller_buffer_canary_past_write(projections, width):
    gate, up, down = projections
    down = replace(down, n=width)
    x = _input()
    backing = torch.full((16 + 3 * width + 64,), -123.0, dtype=torch.float32, device="cuda")
    out = backing[16:16 + 3 * width].view(3, width)
    prefix = _bits(backing[:16]).clone()
    canary = _bits(backing[16 + 3 * width:]).clone()
    pointer = out.data_ptr()
    assert dense_eager(x, gate, up, down, out) is out
    assert out.data_ptr() == pointer and torch.isfinite(out).all()
    reference = torch.empty_like(out)
    assert dense_eager(x, gate, up, down, reference) is reference
    assert torch.equal(_bits(out), _bits(reference))
    assert torch.equal(_bits(backing[:16]), prefix)
    assert torch.equal(_bits(backing[16 + 3 * width:]), canary)


def test_destination_overlapping_input_is_refused(projections, monkeypatch):
    backing = torch.ones((3, 128), dtype=torch.float32, device="cuda")
    # Different dtypes and pointers still overlap the same byte storage.
    x = backing.view(torch.bfloat16)[:, 64:192]
    assert x.shape == backing.shape and x.data_ptr() != backing.data_ptr()
    before = _bits(backing).clone()
    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="overlap"):
        dense_eager(x, *projections, backing)
    assert torch.equal(_bits(backing), before)


def test_destination_overlapping_weight_storage_is_refused(projections, monkeypatch):
    gate, up, down = projections
    out = down.words.view(torch.float32).reshape(-1)[:3 * 128].view(3, 128)
    before = down.words.clone()
    _forbid_work(monkeypatch)
    with pytest.raises(ValueError, match="overlap"):
        dense_eager(_input(), gate, up, down, out)
    assert torch.equal(down.words, before)
