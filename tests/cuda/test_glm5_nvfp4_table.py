"""One synthetic routed layer on the existing packer. No shard and no serve."""

from __future__ import annotations

import pytest
import torch

from tensorfold.families.glm5_next.cuda import nvfp4_table as table

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _proj(n: int, k: int, scale: float, act: float, device: str) -> dict:
    return {
        "weight": torch.full((n, k // 2), 0x11, dtype=torch.uint8, device=device),
        "weight_scale": torch.full((n, k // 16), 0x38, dtype=torch.uint8, device=device),
        "weight_scale_2": torch.tensor(scale, device=device),
        "input_scale": torch.tensor(act, device=device),
    }


def _layer(device: str = "cuda", experts: int = 4, *, unequal: bool = False) -> tuple[dict, dict]:
    scales = {"gate_proj": 1.25, "up_proj": 2.5, "down_proj": 0.5}
    routed = {
        e: {proj: _proj(64, 64, scale * (e + 1) if unequal else scale,
                       0.5 * (e + 1) if unequal else 0.5, device) for proj, scale in scales.items()}
        for e in range(experts)
    }
    shared = {proj: torch.zeros((64, 64), dtype=torch.bfloat16, device=device) for proj in table.PROJECTIONS}
    return routed, shared


@pytest.mark.parametrize("unequal", [False, True], ids=["equal-scales", "unequal-scales"])
@pytest.mark.parametrize("loader", [table.load_synthetic_layer, table.load_checkpoint_layer],
                         ids=["synthetic", "checkpoint"])
def test_e4_pointer_table_aliases_the_owner(unequal, loader):
    routed, shared = _layer(unequal=unequal)
    loaded = loader(layer=3, experts=routed, shared=shared)
    assert loaded.experts == 4 and loaded.combine_weight == 1.0
    assert not hasattr(loaded, "draft_head")
    assert loaded.words_ptr.dtype == torch.int64
    for slot, linear in enumerate(loaded.linears):
        row, col = divmod(slot, 3)
        assert int(loaded.words_ptr[row, col].item()) == linear.words.data_ptr()
        assert int(loaded.bs_ptr[row, col].item()) == linear.bs.data_ptr()
        assert loaded.words_ptr.data_ptr() != linear.words.data_ptr()
        tensors = routed[row][table.PROJECTIONS[col]]
        scale, act = float(tensors["weight_scale_2"]), float(tensors["input_scale"])
        assert (linear.scale, linear.act) == (scale, act)
        assert float(loaded.alpha[row, col]) == table.rounded_alpha(act, scale)
    assert len(set(loaded.alpha[:, 0].tolist())) == (4 if unequal else 1)
    gate, up = loaded.linears[0], loaded.linears[1]
    assert gate.scale == 1.25 and up.scale == 2.5 and gate.act == 0.5
    assert gate.scale != 1.0 / 1.25
    assert table._f32_bits(float(loaded.alpha[0, 0])) == table._f32_bits(table.rounded_alpha(0.5, 1.25))
    assert float(loaded.alpha[0, 0]) != float(loaded.alpha[0, 1])
    shared_ptrs = {t.data_ptr() for t in loaded.shared.values()}
    assert shared_ptrs.isdisjoint(set(loaded.words_ptr.view(-1).tolist()))
    assert loaded.words_ptr.nbytes < gate.words.nbytes


def test_failed_pack_drops_only_the_partial_table():
    routed, shared = _layer()
    caller = routed[0]["gate_proj"]["weight"].data_ptr()
    torch.cuda.synchronize()
    real = table.Fp4Linear.from_checkpoint
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("injected")
        return real(*args, **kwargs)

    warm = real(routed[0]["gate_proj"]["weight"], routed[0]["gate_proj"]["weight_scale"], 1.25, act=0.5)
    del warm
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    table.Fp4Linear.from_checkpoint = boom
    try:
        with pytest.raises(RuntimeError, match="injected"):
            table.load_synthetic_layer(layer=3, experts=routed, shared=shared)
    finally:
        table.Fp4Linear.from_checkpoint = real
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert routed[0]["gate_proj"]["weight"].data_ptr() == caller


def test_weights_nbytes_counts_an_fp4_linear_once():
    from tensorfold.cuda.nvfp4.linear import Fp4Linear, Staging
    from tensorfold.families.glm5_next.cuda.weights import Weights

    staging = Staging()
    staging.w8 = torch.zeros(32, dtype=torch.uint8, device="cuda")
    staging.s8 = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
    a = Fp4Linear(torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32, device="cuda"),
                  torch.zeros((1, 1, 64, 4), dtype=torch.uint8, device="cuda"), 1.25, 64, 64, act=0.5,
                  staging=staging)
    b = Fp4Linear(torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32, device="cuda"),
                  torch.zeros((1, 1, 64, 4), dtype=torch.uint8, device="cuda"), 2.5, 64, 64, act=0.5,
                  staging=staging)
    held = Weights.__new__(Weights)
    held.embed = ()
    held.layers = [a, b]
    held.norm = torch.empty(0, device="cuda")
    held.head = torch.empty(0, device="cuda")
    held.draft_head = None
    held.mtp = None
    assert held.nbytes() == table.owned_nbytes([a, b])
    assert held.nbytes() < a.nbytes() + b.nbytes() + 2 * staging.nbytes()
