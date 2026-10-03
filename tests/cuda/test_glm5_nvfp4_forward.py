"""Packed GLM forward dispatch with tiny synthetic CUDA weights."""

# ruff: noqa: E402 -- CUDA availability MUST be checked before backend imports.

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("GLM NVFP4 forward fixtures need an SM 12 GPU", allow_module_level=True)

from tensorfold.families.glm5_next.cuda import forward, glue, nvfp4_table
from tensorfold.families.glm5_next.cuda.nvfp4_eager import dense_eager, routed_eager
from tensorfold.families.glm5_next.cuda.nvfp4_routed import routed_decode
from tensorfold.families.glm5_next.cuda.weights import LayerW, MoEW


@pytest.fixture(scope="module")
def packed():
    factors = {"gate_proj": (0.03125, 0.375), "up_proj": (0.0625, 0.375),
               "down_proj": (0.015625, 0.625)}
    blocks = {}
    for expert in range(4):
        blocks[expert] = {}
        for column, (proj, (factor, act)) in enumerate(factors.items()):
            rng = torch.Generator().manual_seed(101 + expert * 3 + column)
            blocks[expert][proj] = {
                "weight": torch.randint(0, 256, (128, 64), generator=rng, dtype=torch.uint8).cuda(),
                "weight_scale": torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda"),
                "weight_scale_2": torch.tensor(factor, device="cuda"),
                "input_scale": torch.tensor(act, device="cuda"),
            }
    shared = {proj: torch.full((128, 128), value, dtype=torch.bfloat16, device="cuda")
              for proj, value in zip(nvfp4_table.PROJECTIONS, (0.125, 0.0625, 0.03125))}
    return nvfp4_table.load_synthetic_layer(layer=3, experts=blocks, shared=shared)


@pytest.fixture(autouse=True)
def forbid_grouped(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("a RoutedTable MUST NOT reach grouped expert dispatch")

    for name in ("route", "gate_up", "down"):
        monkeypatch.setattr(forward.grouped, name, fail)


def _fixture(table, rows=1, prefill=False, routed_scale=1.75):
    cfg = SimpleNamespace(top_k=2, experts=4, hidden=128, limit=7.0,
                          routed_scale=routed_scale, norm_topk=True)
    w = SimpleNamespace(cfg=cfg, world=1, comm=None)
    x = torch.linspace(-0.25, 1.75, rows * 128, device="cuda").to(torch.bfloat16).view(rows, 128)
    b = SimpleNamespace(normed=x, prefill=prefill, world=1, plan=None,
                        mlog=torch.empty((rows, 4), device="cuda"),
                        pick=torch.empty((rows, 3), dtype=torch.int32, device="cuda"),
                        wts=torch.empty((rows, 3), device="cuda"),
                        part=torch.full((rows, 128), float("nan"), device="cuda"))
    router = torch.arange(4, device="cuda").to(torch.bfloat16)[:, None].expand(4, 128).contiguous() / 128
    moe = MoEW(router, torch.tensor([0.0, 0.25, -0.25, 0.5], device="cuda"), table)
    return LayerW(3, "dsa", None, None, None, None, moe=moe), w, b


@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("routed_scale", [0.0, 1.75])
def test_one_token_forward_matches_same_table_decode_bits(packed, prefill, routed_scale):
    layer, w, b = _fixture(packed, prefill=prefill, routed_scale=routed_scale)
    result = forward.moe_block(layer, w, b, 1)
    ids = b.pick[:, :w.cfg.top_k].to(torch.int64)
    weights = b.wts[:, :w.cfg.top_k]
    reference = torch.empty_like(b.part)
    routed_decode(b.normed, packed, ids, weights, reference,
                  residual_act=packed.linears[0].act, intermediate_act=packed.linears[2].act, limit=w.cfg.limit)
    assert result.shape == (1, 1, 128) and result.data_ptr() == b.part.data_ptr()
    assert torch.isfinite(result).all() and result.abs().max() > 0
    assert torch.equal(result[0].view(torch.int32), reference.view(torch.int32))
    if routed_scale == 0:
        assert torch.count_nonzero(weights) == 0
        gu = torch.cat([torch.nn.functional.linear(b.normed, packed.shared[proj])
                        for proj in nvfp4_table.PROJECTIONS[:2]], dim=1)
        act = torch.empty_like(b.normed)
        glue.swiglu(gu, act, torch.empty((1, 2), device="cuda"), w.cfg.limit)
        shared = torch.nn.functional.linear(act.float(), packed.shared["down_proj"].float())
        assert shared.abs().max() > 0
        assert torch.equal(result[0].view(torch.int32), shared.view(torch.int32))


@pytest.mark.parametrize("prefill", [False, True])
def test_multirow_forward_stays_on_same_eager_backend(packed, prefill, monkeypatch):
    layer, w, b = _fixture(packed, rows=3, prefill=prefill)

    def fail(*args, **kwargs):
        pytest.fail("multirow execution MUST remain eager")

    monkeypatch.setattr(forward, "routed_decode", fail)
    result = forward.moe_block(layer, w, b, 3)
    experts = [tuple(packed.linears[i:i + 3]) for i in range(0, len(packed.linears), 3)]
    reference = torch.empty_like(b.part)
    routed_eager(b.normed, experts, packed.shared, b.pick[:, :2].to(torch.int64), b.wts[:, :2], reference,
                 backend="prompt" if prefill else "lane", limit=w.cfg.limit)
    assert torch.isfinite(result).all()
    assert torch.equal(result[0].view(torch.int32), reference.view(torch.int32))


@pytest.mark.parametrize("rows,prefill", [(1, False), (3, True)])
def test_packed_dense_forward_matches_dense_eager(packed, rows, prefill):
    layer, w, b = _fixture(packed, rows=rows, prefill=prefill)
    dense = replace(packed, linears=packed.linears[:3], shared={})
    layer = replace(layer, moe=None, dense_nvfp4=dense)
    result = forward.mlp_block(layer, w, b, rows)
    reference = torch.empty_like(b.part)
    dense_eager(b.normed, *dense.linears, reference, limit=w.cfg.limit)
    assert result.data_ptr() == b.part.data_ptr() and torch.isfinite(result).all()
    assert torch.equal(result[0].view(torch.int32), reference.view(torch.int32))


def test_static_factor_disagreement_is_refused(packed):
    owners = list(packed.linears)
    owners[4] = replace(owners[4], act=owners[4].act * 2)
    layer, w, b = _fixture(replace(packed, linears=owners))
    with pytest.raises(ValueError, match="common static"):
        forward.moe_block(layer, w, b, 1)
