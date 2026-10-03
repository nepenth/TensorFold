"""CPU checks for the routed-expert table, with a stub packer and no kernel."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tensorfold.cuda.nvfp4.linear import Fp4Linear, Staging
from tensorfold.families.glm5_next.cuda import nvfp4_table as table


def test_layer_45_raises_before_any_expert_is_read():
    with pytest.raises(ValueError, match="layer 45 is not loaded"):
        table.load_synthetic_layer(layer=45, experts={0: {}}, shared={})


def test_missing_weight_scale_raises_before_packing():
    experts = {0: {"gate_proj": {"weight": object(), "weight_scale": object(), "input_scale": object()}}}
    with pytest.raises(ValueError, match="missing weight_scale_2"):
        table.load_synthetic_layer(layer=3, experts=experts, shared={})


def _layer(scales: tuple, acts: tuple) -> tuple[dict, dict]:
    experts = {e: {proj: {
        "weight": object(), "weight_scale": object(),
        "weight_scale_2": torch.tensor(scale * (col + 1)), "input_scale": torch.tensor(act),
    } for col, proj in enumerate(table.PROJECTIONS)}
        for e, (scale, act) in enumerate(zip(scales, acts))}
    shared = {proj: torch.zeros(1, dtype=torch.bfloat16) for proj in table.PROJECTIONS}
    return experts, shared


UNEQUAL_SCALES = (5.8128720411332324e-05, 4.650297705666162e-05, 4.3596541217993945e-05)


@pytest.mark.parametrize("scales,acts", [
    ((1.25,) * 3, (0.5,) * 3),
    (UNEQUAL_SCALES, (0.5,) * 3),
    ((1.25,) * 3, (0.5, 0.75, 1.0)),
    (UNEQUAL_SCALES, (0.5, 0.75, 1.0)),
], ids=["equal", "unequal-weight", "unequal-input", "unequal-both"])
@pytest.mark.parametrize("checkpoint", [False, True], ids=["synthetic", "checkpoint"])
def test_each_expert_keeps_its_scales_and_alpha(monkeypatch, scales, acts, checkpoint):
    def pack(weight, weight_scale, scale, *, act):
        return Fp4Linear(torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32),
                         torch.zeros((1, 1, 64, 4), dtype=torch.uint8), scale, 64, 64, act=act)

    monkeypatch.setattr(Fp4Linear, "from_checkpoint", pack)
    experts, shared = _layer(scales, acts)
    if checkpoint:
        loaded = table.load_checkpoint_layer(layer=3, experts=experts, shared=shared)
    else:
        # Same-expert replicas MUST agree even when the experts differ from each other.
        loaded = table.load_synthetic_layer(layer=3, experts=experts, shared=shared,
                                           shards=[dict(reversed(list(experts.items())))])
    assert loaded.experts == 3 and len(loaded.linears) == 9
    for slot, linear in enumerate(loaded.linears):
        e, col = divmod(slot, 3)
        tensors = experts[e][table.PROJECTIONS[col]]
        scale, act = float(tensors["weight_scale_2"]), float(tensors["input_scale"])
        assert (linear.scale, linear.act) == (scale, act)
        assert float(loaded.alpha[e, col]) == table.rounded_alpha(act, scale)
    for col in range(3):
        distinct = len(set(loaded.alpha[:, col].tolist()))
        assert distinct == (1 if scales[0] == scales[1] and acts[0] == acts[1] else 3)


@pytest.mark.parametrize("expert", [0, 2])
@pytest.mark.parametrize("proj", table.PROJECTIONS)
@pytest.mark.parametrize("key", ["weight_scale_2", "input_scale"])
def test_shard_disagreement_stops_the_load(monkeypatch, expert, proj, key):
    primary, shared = _layer(UNEQUAL_SCALES, (0.5, 0.75, 1.0))
    other, _ = _layer(UNEQUAL_SCALES, (0.5, 0.75, 1.0))
    other[expert][proj][key] *= 2

    def forbidden(*args, **kwargs):
        pytest.fail("same-expert shard disagreement MUST raise before packing")

    monkeypatch.setattr(Fp4Linear, "from_checkpoint", forbidden)
    with pytest.raises(ValueError, match=rf"experts\.{expert}\.{proj} {key} disagrees across shards"):
        table.load_synthetic_layer(layer=3, experts=primary, shared=shared, shards=[other])


def test_shared_expert_must_be_bf16_weight():
    experts = {0: {proj: {
        "weight": object(), "weight_scale": object(),
        "weight_scale_2": torch.tensor(1.0), "input_scale": torch.tensor(0.5),
    } for proj in table.PROJECTIONS}}
    shared = {proj: torch.zeros(1, dtype=torch.uint8) for proj in table.PROJECTIONS}
    with pytest.raises(ValueError, match="shared expert gate_proj is BF16"):
        table.load_synthetic_layer(layer=3, experts=experts, shared=shared)


def test_prefetch_is_routed_keys_and_shared_weight_only():
    names = table.prefetch_names(3, 4)
    assert len(names) == 4 * 3 * 4 + 3
    assert all(not name.endswith("shared_experts.gate_proj.weight_scale") for name in names)
    assert names[-3:] == [
        "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.up_proj.weight",
        "model.language_model.layers.3.mlp.shared_experts.down_proj.weight",
    ]
    with pytest.raises(ValueError, match="layer 45"):
        table.prefetch_names(45, 1)


def test_bf16_arms_and_refusals():
    assert table.storage_arm("model.language_model.layers.3.mlp.gate.weight") == "bf16"
    assert table.storage_arm("model.language_model.embed_tokens.weight") == "bf16"
    assert table.storage_arm("lm_head.weight") == "bf16"
    assert table.storage_arm("model.language_model.layers.3.self_attn.o_proj.weight") == "bf16"
    assert table.storage_arm("model.language_model.layers.3.input_layernorm.weight") == "bf16"
    assert table.storage_arm("model.language_model.layers.3.mlp.shared_experts.gate_proj.weight") == "bf16-weight"
    assert table.storage_arm("model.language_model.layers.3.mlp.experts.0.gate_proj.weight") == "nvfp4"
    assert table.storage_arm("model.visual.proj") == "skip"
    with pytest.raises(ValueError, match="layer 45"):
        table.storage_arm("model.language_model.layers.45.mlp.experts.0.gate_proj.weight")
    with pytest.raises(ValueError, match="draft head"):
        table.storage_arm("model.draft_head.weight")


def test_e288_addresses_do_not_allocate_production_weights():
    slots = table.empty_address_table(288)
    tiny = [torch.empty(1, dtype=torch.uint8) for _ in range(slots.numel())]
    table.bind_addresses(slots, tiny)
    assert slots.shape == (288, 3)
    assert slots.numel() * slots.element_size() == 288 * 3 * 8
    assert sum(t.nbytes for t in tiny) == 288 * 3
    assert sum(t.nbytes for t in tiny) < 2048 * 2048
    assert int(slots.view(-1)[0].item()) == tiny[0].data_ptr()
    assert int(slots.view(-1)[-1].item()) == tiny[-1].data_ptr()


def test_shared_staging_is_not_double_counted():
    staging = Staging()
    staging.w8 = torch.zeros(32, dtype=torch.uint8)
    staging.s8 = torch.zeros(16, dtype=torch.bfloat16)
    a = Fp4Linear(torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32), torch.zeros((1, 1, 64, 4), dtype=torch.uint8),
                  1.25, 64, 64, act=0.5, staging=staging)
    b = Fp4Linear(torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32), torch.zeros((1, 1, 64, 4), dtype=torch.uint8),
                  2.5, 64, 64, act=0.5, staging=staging)
    once = table.owned_nbytes([a, b])
    assert once == a.words.nbytes + a.bs.nbytes + b.words.nbytes + b.bs.nbytes + staging.nbytes()
    assert once < a.nbytes() + b.nbytes() + 2 * staging.nbytes()


def test_production_shaped_table_is_refused_before_packing():
    weight = torch.empty((2048, 1024), dtype=torch.uint8)
    block = {proj: {
        "weight": weight, "weight_scale": object(),
        "weight_scale_2": torch.tensor(1.0), "input_scale": torch.tensor(0.5),
    } for proj in table.PROJECTIONS}
    experts = {i: block for i in range(32)}
    shared = {proj: torch.zeros(1, dtype=torch.bfloat16) for proj in table.PROJECTIONS}
    with pytest.raises(ValueError, match="refuses production"):
        table.load_synthetic_layer(layer=3, experts=experts, shared=shared)


def test_loader_does_not_call_grouped_experts():
    text = Path(table.__file__).read_text()
    assert "cuda.experts" not in text
    assert "nvfp4.experts" not in text
    assert "matmul_group" not in text
