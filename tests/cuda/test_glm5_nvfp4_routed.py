"""Routed eager oracle. Each down reads its own intermediate. No shard pull."""

# ruff: noqa: E402 -- CUDA availability MUST be checked before backend imports.

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("GLM NVFP4 eager oracle needs an SM 12 GPU", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear
from tensorfold.families.glm5_next.cuda import glue
from tensorfold.families.glm5_next.cuda.nvfp4_eager import routed_eager


def _linear(code: int, factor: float, act: float = 0.25) -> Fp4Linear:
    return Fp4Linear.from_checkpoint(
        torch.full((128, 64), code, dtype=torch.uint8, device="cuda"),
        torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda"),
        factor, act=act)


def _experts():
    codes = (0x11, 0x22, 0x24, 0x26)
    factors = (0.03125, 0.0625, 0.125, 0.25)
    built = []
    for code, factor in zip(codes, factors):
        built.append((_linear(code, factor, 0.25), _linear(code ^ 0x02, factor * 2, 0.25),
                      _linear(code ^ 0x04, factor / 2, 0.125)))
    return tuple(built)


def _shared():
    gate = torch.full((128, 128), 0.25, dtype=torch.bfloat16, device="cuda")
    up = torch.full((128, 128), 0.125, dtype=torch.bfloat16, device="cuda")
    down = torch.full((128, 128), 0.5, dtype=torch.bfloat16, device="cuda")
    return {"gate_proj": gate, "up_proj": up, "down_proj": down}


def _input(rows=3):
    rng = torch.Generator().manual_seed(91)
    return torch.randn((rows, 128), generator=rng).to(torch.bfloat16).cuda()


def _ids(rows=3):
    return torch.tensor([[3, 0], [1, 3], [2, 1]], dtype=torch.int64, device="cuda")[:rows]


def _weights(rows=3):
    return torch.tensor([[0.5, 0.25], [1.0, 0.0], [0.125, 0.75]], dtype=torch.float32, device="cuda")[:rows]


def _out(rows=3):
    return torch.full((rows, 128), float("nan"), dtype=torch.float32, device="cuda")


def _bits(x):
    return x.view(torch.int32)


def _shared_only(x, shared, limit=10.0):
    gu = torch.cat((
        torch.nn.functional.linear(x, shared["gate_proj"]),
        torch.nn.functional.linear(x, shared["up_proj"]),
    ), dim=1)
    activated = torch.empty((x.shape[0], 128), dtype=torch.bfloat16, device="cuda")
    sums = torch.empty((x.shape[0], 2), dtype=torch.float32, device="cuda")
    glue.swiglu(gu, activated, sums, limit)
    return torch.nn.functional.linear(activated.float(), shared["down_proj"].float())


def _one(row, gate, up, down, backend, limit=10.0):
    prompt_rows = backend == "prompt"
    grouped = checkpoint.matmul_group(row, [gate, up], prompt_rows=prompt_rows)
    gu = torch.cat(grouped, dim=1)
    activated = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda")
    sums = torch.empty((1, 2), dtype=torch.float32, device="cuda")
    glue.swiglu(gu, activated, sums, limit)
    slot = torch.empty((1, 128), dtype=torch.float32, device="cuda")
    if prompt_rows:
        checkpoint.prompt(checkpoint.A4, activated, down, out=slot, f32=True)
    else:
        checkpoint.matmul(checkpoint.A4, activated, down, out=slot, f32=True)
    return slot


def _compose(x, experts, shared, ids, weights, backend):
    rows, slots = ids.shape
    ey = torch.empty((rows, slots + 1, 128), dtype=torch.float32, device="cuda")
    for row in range(rows):
        for slot in range(slots):
            gate, up, down = experts[int(ids[row, slot])]
            ey[row, slot] = _one(x[row:row + 1], gate, up, down, backend)[0]
    ey[:, slots] = _shared_only(x, shared)
    combined = torch.empty((rows, slots + 1), dtype=torch.float32, device="cuda")
    combined[:, :slots] = weights
    combined[:, slots] = 1.0
    out = torch.empty((rows, 128), dtype=torch.float32, device="cuda")
    glue.combine(ey, combined, out)
    return out


@pytest.mark.parametrize("backend", ["lane", "prompt"])
def test_routed_repeat_matches_same_backend_composition(backend):
    experts, shared = _experts(), _shared()
    assert experts[0][0].scale != experts[3][0].scale
    assert not torch.equal(experts[0][0].words, experts[3][0].words)
    x, ids, weights = _input(), _ids(), _weights()
    first, second = _out(), _out()
    assert routed_eager(x, experts, shared, ids, weights, first, backend=backend) is first
    assert routed_eager(x, experts, shared, ids, weights, second, backend=backend) is second
    assert torch.isfinite(first).all() and first.abs().max() > 0
    assert torch.equal(_bits(first), _bits(second))
    assert torch.equal(_bits(first), _bits(_compose(x, experts, shared, ids, weights, backend)))


def test_lane_and_prompt_are_not_compared():
    experts, shared = _experts(), _shared()
    x, ids, weights = _input(), _ids(), _weights()
    lane, prompt = _out(), _out()
    routed_eager(x, experts, shared, ids, weights, lane, backend="lane")
    routed_eager(x, experts, shared, ids, weights, prompt, backend="prompt")
    assert torch.equal(_bits(lane), _bits(_compose(x, experts, shared, ids, weights, "lane")))
    assert torch.equal(_bits(prompt), _bits(_compose(x, experts, shared, ids, weights, "prompt")))


def test_downs_are_per_expert_and_not_grouped(monkeypatch):
    experts, shared = _experts(), _shared()
    x, ids, weights = _input(1), _ids(1), _weights(1)
    groups, downs = [], []
    grouped, matmul, prompt = checkpoint.matmul_group, checkpoint.matmul, checkpoint.prompt

    def record_group(values, lins, prompt_rows=False, outs=None, tile=0):
        groups.append(lins)
        return grouped(values, lins, prompt_rows, outs, tile)

    def record_down(mode, values, lin, out=None, f32=False, tile=0):
        downs.append(lin)
        return matmul(mode, values, lin, out, f32)

    def record_prompt(mode, values, lin, out=None, f32=False, tile=0):
        downs.append(lin)
        return prompt(mode, values, lin, out, f32, tile)

    monkeypatch.setattr(checkpoint, "matmul_group", record_group)
    monkeypatch.setattr(checkpoint, "matmul", record_down)
    monkeypatch.setattr(checkpoint, "prompt", record_prompt)
    routed_eager(x, experts, shared, ids, weights, _out(1), backend="lane")
    down_ids = [id(item[2]) for item in experts]
    assert groups and all(len(pair) == 2 for pair in groups)
    assert all(id(pair[0]) not in down_ids and id(pair[1]) not in down_ids for pair in groups)
    assert downs and all(id(lin) in down_ids for lin in downs)


def test_highest_expert_id_is_not_the_shared_slot():
    experts, shared = _experts(), _shared()
    x = _input(1)
    high = torch.tensor([[3]], dtype=torch.int64, device="cuda")
    low = torch.tensor([[0]], dtype=torch.int64, device="cuda")
    weight = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    highest, other = _out(1), _out(1)
    routed_eager(x, experts, shared, high, weight, highest, backend="lane")
    routed_eager(x, experts, shared, low, weight, other, backend="lane")
    assert not torch.equal(_bits(highest), _bits(other))
    assert torch.equal(_bits(highest), _bits(_compose(x, experts, shared, high, weight, "lane")))


def test_zero_routed_weights_still_contribute_shared():
    experts, shared = _experts(), _shared()
    x = _input()
    ids = _ids()
    zeros = torch.zeros_like(_weights())
    out = _out()
    routed_eager(x, experts, shared, ids, zeros, out, backend="prompt")
    shared_only = _shared_only(x, shared)
    assert shared_only.abs().max() > 0
    assert torch.equal(_bits(out), _bits(shared_only))


def test_shared_combine_weight_is_one(monkeypatch):
    experts, shared = _experts(), _shared()
    seen = {}
    real = glue.combine

    def record(y, wts, out):
        seen["wts"] = wts.detach().clone()
        return real(y, wts, out)

    monkeypatch.setattr(glue, "combine", record)
    routed_eager(_input(1), experts, shared, _ids(1), _weights(1), _out(1), backend="lane")
    assert seen["wts"].shape[1] == 3
    assert torch.equal(seen["wts"][:, -1], torch.ones(1, device="cuda"))
    assert torch.equal(seen["wts"][:, :2], _weights(1))


def test_refused_destination_is_unchanged(monkeypatch):
    experts, shared = _experts(), _shared()
    backing = torch.full((3, 256), -123.0, dtype=torch.float32, device="cuda")
    out = backing[:, ::2]
    before = backing.clone()

    def fail(*args, **kwargs):
        pytest.fail("refusal must precede quantization, multiply, and SwiGLU")

    for name in ("quant4", "matmul_group", "matmul", "prompt", "_lane"):
        monkeypatch.setattr(checkpoint, name, fail)
    monkeypatch.setattr(glue, "swiglu", fail)
    monkeypatch.setattr(glue, "combine", fail)
    with pytest.raises(ValueError, match="contiguous"):
        routed_eager(_input(), experts, shared, _ids(), _weights(), out, backend="lane")
    assert torch.equal(backing, before)


@pytest.mark.parametrize("backend", ["lane", "prompt"])
@pytest.mark.parametrize("expert_index", [0, 3])
@pytest.mark.parametrize("n,k", [(256, 128), (128, 256)])
def test_mismatched_up_dimensions_are_refused_before_work(monkeypatch, backend, expert_index, n, k):
    experts, shared = list(_experts()), _shared()
    gate, _, down = experts[expert_index]
    up = Fp4Linear.from_checkpoint(
        torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda"),
        torch.full((n, k // 16), 0x38, dtype=torch.uint8, device="cuda"),
        0.0625, act=gate.act)
    experts[expert_index] = (gate, up, down)
    x, ids, weights, out = _input(), _ids(), _weights(), _out()
    before = _bits(out).clone()

    def fail(*args, **kwargs):
        pytest.fail("refusal must precede quantization, multiply, and SwiGLU")

    for name in ("quant4", "matmul_group", "matmul", "prompt", "_lane", "_gemm"):
        monkeypatch.setattr(checkpoint, name, fail)
    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    monkeypatch.setattr(glue, "swiglu", fail)
    monkeypatch.setattr(glue, "combine", fail)
    with pytest.raises(ValueError, match="gate/up dimensions must match"):
        routed_eager(x, experts, shared, ids, weights, out, backend=backend)
    assert torch.equal(_bits(out), before)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_shared_device_mismatch_is_refused_before_work(monkeypatch, projection):
    experts, shared = _experts(), _shared()
    other_device = (shared[projection].device.index + 1) % torch.cuda.device_count()
    shared[projection] = shared[projection].to(f"cuda:{other_device}")
    x, ids, weights, out = _input(), _ids(), _weights(), _out()
    before = _bits(out).clone()

    def fail(*args, **kwargs):
        pytest.fail("refusal must precede quantization, multiply, and SwiGLU")

    for name in ("quant4", "matmul_group", "matmul", "prompt", "_lane", "_gemm"):
        monkeypatch.setattr(checkpoint, name, fail)
    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    monkeypatch.setattr(glue, "swiglu", fail)
    monkeypatch.setattr(glue, "combine", fail)
    with pytest.raises(ValueError, match="projection weights must be on the input device"):
        routed_eager(x, experts, shared, ids, weights, out, backend="lane")
    assert torch.equal(_bits(out), before)


def test_canary_past_the_write():
    experts, shared = _experts(), _shared()
    x = _input()
    backing = torch.full((16 + 3 * 128 + 64,), -123.0, dtype=torch.float32, device="cuda")
    out = backing[16:16 + 3 * 128].view(3, 128)
    prefix = _bits(backing[:16]).clone()
    canary = _bits(backing[16 + 3 * 128:]).clone()
    assert routed_eager(x, experts, shared, _ids(), _weights(), out, backend="lane") is out
    assert torch.isfinite(out).all()
    assert torch.equal(_bits(backing[:16]), prefix)
    assert torch.equal(_bits(backing[16 + 3 * 128:]), canary)


def test_overlap_with_shared_storage_is_refused(monkeypatch):
    experts, shared = _experts(), _shared()
    out = shared["down_proj"].view(torch.float32).reshape(-1)[:3 * 128].view(3, 128)
    before = shared["down_proj"].clone()

    def fail(*args, **kwargs):
        pytest.fail("refusal must precede work")

    monkeypatch.setattr(checkpoint, "matmul_group", fail)
    monkeypatch.setattr(glue, "swiglu", fail)
    with pytest.raises(ValueError, match="overlap"):
        routed_eager(_input(), experts, shared, _ids(), _weights(), out, backend="prompt")
    assert torch.equal(shared["down_proj"], before)
