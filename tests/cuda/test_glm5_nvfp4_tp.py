"""Task 9A: synthetic TP=2 on one Spark, existing lane only, no collectives.

Numeric errors are recorded under a named, unfrozen envelope. A successful
structural assertion MUST NOT be interpreted as numeric envelope qualification.
"""

# ruff: noqa: E402 -- Backend imports follow the optional PyTorch import.

from __future__ import annotations

from dataclasses import asdict
import json
import math

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4 import checkpoint, linear
from tensorfold.families.glm5_next.cuda import split
from tensorfold.families.glm5_next.cuda.nvfp4_tp import TP_ENVELOPE, synthetic_tp

cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="Task 9A lane arithmetic needs one SM 12 GPU",
)


def _source(n, k, seed=901, factor=0.03125):
    rng = torch.Generator().manual_seed(seed)
    # Asymmetric halves and finite E4M3 scales expose reversed rank ordering.
    weight = torch.randint(0, 256, (n, k // 2), generator=rng, dtype=torch.uint8)
    scale_codes = torch.tensor([0x28, 0x30, 0x38, 0x40], dtype=torch.uint8)
    scales = scale_codes[torch.randint(0, 4, (n, k // 16), generator=rng)]
    return {"weight": weight, "weight_scale": scales,
            "weight_scale_2": torch.tensor(factor, dtype=torch.float32),
            "input_scale": torch.tensor(0.25, dtype=torch.float32)}


def _input(rows, k):
    rng = torch.Generator().manual_seed(902)
    return torch.randn((rows, k), generator=rng).to(torch.bfloat16).to("cuda:0")


def _bits(x):
    assert x.dtype == torch.float32
    return x.contiguous().view(torch.int32)


def _record(result, record_property):
    observation = asdict(result.record)
    observation["max_abs_error"] = result.max_abs_error
    record_property("tp_observation", json.dumps(observation, sort_keys=True))
    assert result.record.envelope == TP_ENVELOPE
    assert result.record.envelope_frozen is False
    assert result.record.rank_order == (0, 1)
    # Finiteness and layout are structural checks, with no numeric error bound.
    assert torch.isfinite(result.full).all() and torch.isfinite(result.combined).all()
    assert all(out.dtype == torch.float32 and torch.isfinite(out).all() for out in result.rank_outputs)
    assert math.isfinite(result.max_abs_error)
    assert result.max_abs_error == float((result.combined - result.full).abs().max())


def _forbid_launch(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("refusal MUST precede every pack, quantization, and multiply launch")

    for name in ("_ext", "pack4", "quant4", "matmul", "_lane"):
        monkeypatch.setattr(checkpoint, name, fail)
    monkeypatch.setattr(linear, "_ext", fail)  # experts.cu MUST NEVER be called.


def test_packed_width_96_refused_by_existing_guard_before_launch(monkeypatch):
    _forbid_launch(monkeypatch)
    calls = []
    legal = split.nvfp4_column_legal

    def record(shape, dtype):
        calls.append((shape, dtype))
        return legal(shape, dtype)

    monkeypatch.setattr(split, "nvfp4_column_legal", record)
    with pytest.raises(ValueError, match="packed width 96.*multiple of 64"):
        synthetic_tp("model.layers.0.mlp.down_proj", _source(128, 192))
    assert calls == [([128, 96], "U8")]


@pytest.mark.parametrize("projection,n,k,match", [
    ("gate", 127, 128, "even N"),
    ("up", 128, 96, "multiple of 64"),
    ("down", 128, 64, "multiple of 64"),
])
def test_illegal_rank_shapes_refused_before_launch(monkeypatch, projection, n, k, match):
    _forbid_launch(monkeypatch)
    with pytest.raises(ValueError, match=match):
        synthetic_tp(f"model.layers.0.mlp.{projection}_proj", _source(n, k))


@cuda
@pytest.mark.parametrize("projection,factor", [("gate", 0.03125), ("up", 0.0625)])
@pytest.mark.parametrize("n,k,full_sk,rank_sk", [
    (128, 128, 1, 1),
    (128, 1024, 2, 2),
    (2048, 4096, 8, 8),
    (6144, 4096, 2, 4),
    (12288, 4096, 1, 2),
])
def test_row_split_rank_order_shapes_and_split_k(projection, factor, n, k, full_sk, rank_sk,
                                                monkeypatch, record_property):
    # The alternate linear extension includes experts.cu and is outside Task 9A.
    monkeypatch.setattr(linear, "_ext", lambda: pytest.fail("experts.cu is forbidden"))
    name = f"model.layers.0.mlp.{projection}_proj"
    assert split.rule(name + ".weight") == split.rule(name + ".weight_scale") == "row"
    owners = synthetic_tp(name, _source(n, k, factor=factor), device="cuda:0")
    record = owners.record
    assert record.kind == "row"
    assert record.full_shape == (n, k) and record.rank_shapes == ((n // 2, k), (n // 2, k))
    assert record.full_split_k == full_sk == qmm.split_k(n, k)
    assert record.rank_split_k == (rank_sk, rank_sk)
    for rank, lin in enumerate(owners.ranks):
        assert record.rank_split_k[rank] == qmm.split_k(lin.n, lin.k)
        assert lin.k % 64 == 0 and lin.words.device == owners.full.words.device
        assert lin.act == owners.full.act == 0.25 and lin.scale == owners.full.scale == factor
        # Exact tile-aligned packed-row subsets, including the stored scale bits.
        tiles = n // 2 // 64
        assert torch.equal(lin.words, owners.full.words[rank * tiles:(rank + 1) * tiles])
        assert torch.equal(lin.bs, owners.full.bs[rank * tiles:(rank + 1) * tiles])
    result = owners.project(_input(3, k))
    _record(result, record_property)
    assert result.full.shape == result.combined.shape == (3, n)
    assert all(out.shape == (3, n // 2) for out in result.rank_outputs)
    assert torch.equal(_bits(result.combined), _bits(torch.cat(result.rank_outputs, dim=1)))
    assert not torch.equal(_bits(result.combined), _bits(torch.cat(result.rank_outputs[::-1], dim=1)))
    if full_sk == rank_sk:
        # Only exact packed-row subsets with the same K reduction permit bitwise comparison.
        assert torch.equal(_bits(result.full), _bits(result.combined))
    # Differing split_k records max_abs_error; no bound or equality assertion is applied.


@cuda
@pytest.mark.parametrize("n,k,full_sk,rank_sk", [(128, 256, 1, 1), (128, 2048, 4, 2), (4096, 2048, 4, 2)])
def test_down_column_split_fp32_partials_sum_rank_zero_first(n, k, full_sk, rank_sk,
                                                          monkeypatch, record_property):
    monkeypatch.setattr(linear, "_ext", lambda: pytest.fail("experts.cu is forbidden"))
    name = "model.layers.0.mlp.down_proj"
    assert split.rule(name + ".weight") == split.rule(name + ".weight_scale") == "col"
    owners = synthetic_tp(name, _source(n, k), device="cuda:0")
    record = owners.record
    assert record.kind == "col"
    assert record.full_shape == (n, k) and record.rank_shapes == ((n, k // 2), (n, k // 2))
    assert record.full_split_k == full_sk == qmm.split_k(n, k)
    assert record.rank_split_k == (rank_sk, rank_sk)
    for rank, lin in enumerate(owners.ranks):
        assert record.rank_split_k[rank] == qmm.split_k(lin.n, lin.k)
        assert lin.k % 64 == 0 and lin.words.device == owners.full.words.device
        assert lin.act == owners.full.act and lin.scale == owners.full.scale
        groups = k // 2 // 64
        assert torch.equal(lin.words, owners.full.words[:, rank * groups:(rank + 1) * groups])
        assert torch.equal(lin.bs, owners.full.bs[:, rank * groups:(rank + 1) * groups])
    x = _input(3, k)
    calls, additions = [], []
    matmul, add = checkpoint.matmul, torch.add

    def record_lane(mode, values, lin, **kwargs):
        assert mode == checkpoint.A4 and kwargs == {"f32": True}
        calls.append(lin)
        return matmul(mode, values, lin, **kwargs)

    def record_add(first, second):
        additions.append((first, second))
        return add(first, second)

    with monkeypatch.context() as tracing:
        tracing.setattr(checkpoint, "matmul", record_lane)
        tracing.setattr(torch, "add", record_add)
        result = owners.project(x)
    assert len(calls) == 3
    assert all(actual is expected for actual, expected in zip(calls, (owners.full, *owners.ranks)))
    assert len(additions) == 1
    assert additions[0][0] is result.rank_outputs[0] and additions[0][1] is result.rank_outputs[1]
    _record(result, record_property)
    assert result.full.shape == result.combined.shape == (3, n)
    for rank, partial in enumerate(result.rank_outputs):
        assert partial.shape == (3, n) and partial.dtype == torch.float32
        # Independent lane invocation verifies the weight/activation halves stay paired by rank.
        reference = checkpoint.matmul(checkpoint.A4, x[:, rank * (k // 2):(rank + 1) * (k // 2)].contiguous(),
                                      owners.ranks[rank], f32=True)
        assert torch.equal(_bits(partial), _bits(reference))
    first, second = result.rank_outputs
    assert not torch.equal(_bits(first), _bits(second))
    assert torch.equal(_bits(result.combined), _bits(first + second))
    rounded_partials = first.to(torch.bfloat16).float() + second.to(torch.bfloat16).float()
    assert not torch.equal(_bits(result.combined), _bits(rounded_partials))
    # Even matching split_k with a column split does not license full-vs-TP bitwise equality.
