"""CPU checks for the NVIDIA GLM NVFP4 config and startup policy. No weights and no GPU.

``tests/cuda/`` is not collected unless PyTorch sees an NVIDIA GPU, so this file stays next to the other CPU GLM tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfold.families import glm5_next
from tensorfold.families.glm5_next.cuda.nvfp4_policy import decide

NVIDIA_QUANT = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "config_groups": {
        "group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"],
        }
    },
    "ignore": ["lm_head", "model.language_model.embed_tokens"],
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    "producer": {"name": "modelopt", "version": "0.47.0.dev393+ga4bc45b30.d20260828"},
}

TEXT = {
    "hidden_size": 4096,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "rms_norm_eps": 1e-5,
    "num_attention_heads": 8,
    "q_lora_rank": 16,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "v_head_dim": 16,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "intermediate_size": 64,
    "routed_scaling_factor": 2.5,
    "eos_token_id": 1,
    "layer_types": ["linear_attention", "deepseek_sparse_attention"],
    "num_nextn_predict_layers": 1,
}


def _family():
    from tensorfold.families import Family

    return Family("glm5_next", glm5_next.TITLE, glm5_next.__name__, True)


def _config(**extra) -> dict:
    return {"model_type": "glm5_next", "text_config": TEXT, "quantization_config": NVIDIA_QUANT, **extra}


def test_modelopt_block_is_recognized():
    from tensorfold.cuda.nvfp4.format import config_block, scheme

    block = config_block(_config())
    assert block is not None
    assert block["quant_method"] == "modelopt"
    assert block["quant_algo"] == "NVFP4"
    assert block["config_groups"]["group_0"]["weights"]["group_size"] == 16
    assert scheme({"weight": ("U8", [2048, 2048]), "weight_scale": ("F8_E4M3", [2048, 256])}) == "nvfp4"


def test_require_readable_still_refuses_modelopt():
    from tensorfold.families import require_readable

    with pytest.raises(ValueError, match="does not read this checkpoint's weights"):
        require_readable(_family(), _config(), "cuda")


def test_family_check_still_refuses_modelopt(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    with pytest.raises(ValueError, match="modelopt NVFP4"):
        glm5_next.check(tmp_path)


def test_glm_cuda_loader_still_refuses_modelopt(tmp_path: Path):
    """The refusal is the quant gate, not a missing field earlier in Config.read."""

    pytest.importorskip("torch")
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    from tensorfold.families.glm5_next.cuda.weights import load

    with pytest.raises(ValueError, match="MLX 4-bit or EXL3"):
        load(tmp_path, rank=0)


@pytest.mark.parametrize("raw", [None, "", "0", "auto", " AUTO "])
def test_serial_nvfp4_leaves_mtp_off(raw):
    assert decide(raw, no_drafts=True, drafter=False, parallel=False) == "serial"


def test_explicit_mtp_is_not_the_unset_default():
    with pytest.raises(ValueError, match="unqualified"):
        decide("1", no_drafts=True, drafter=False, parallel=False)
    assert decide(None, no_drafts=True, drafter=False, parallel=False) == "serial"


@pytest.mark.parametrize("kwargs", [
    {"no_drafts": False, "drafter": False, "parallel": False},
    {"no_drafts": True, "drafter": True, "parallel": False},
    {"no_drafts": True, "drafter": False, "parallel": True},
])
def test_drafts_and_parallel_are_refused(kwargs):
    with pytest.raises(ValueError, match="unqualified|does not serve"):
        decide("0", **kwargs)


def test_bad_mtp_value_is_named():
    with pytest.raises(ValueError, match="TF_GLM_MTP"):
        decide("yes", no_drafts=True, drafter=False, parallel=False)
