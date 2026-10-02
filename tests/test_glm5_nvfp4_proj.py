"""CPU checks for one GLM dense NVFP4 projection. No packing and no kernel."""

from __future__ import annotations

import pytest

from tensorfold.families.glm5_next.cuda.nvfp4_proj import dense_projection


def test_vision_is_skipped_without_building_a_linear():
    assert dense_projection("model.visual.proj", {}) is None


def test_routed_expert_names_are_refused():
    with pytest.raises(ValueError, match="routed experts are not wired"):
        dense_projection("model.layers.3.mlp.experts.0.gate_proj.weight", {})


def test_missing_weight_scale_raises_before_packing():
    with pytest.raises(ValueError, match="missing weight_scale_2"):
        dense_projection("model.layers.0.mlp.gate_proj", {
            "weight": object(),
            "weight_scale": object(),
            "input_scale": object(),
        })
