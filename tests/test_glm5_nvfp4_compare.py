"""CPU tests for the GLM NVFP4 manifest and the offline comparator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "docs" / "plans" / "notes"


def _load(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare = _load("glm_nvfp4_compare")
record = _load("glm_nvfp4_record")


def test_pinned_manifest_loads():
    data = compare.load_manifest(NOTES / "nvidia-glm-nvfp4-manifest.json")
    policy = json.loads((NOTES / "nvidia-glm-nvfp4-numeric-policy.json").read_text())
    assert data["world_size"] == 2
    assert data["reference_status"] == "BLOCKED_REFERENCE"
    assert data["numeric_policy_id"] == policy["id"]
    assert data["checkpoint_revision"] == "da920bb0b9f4a06727223a349e55468e38352348"


def test_manifest_rejects_a_bad_world_and_a_hash_mismatch(tmp_path: Path):
    source = json.loads((NOTES / "nvidia-glm-nvfp4-manifest.json").read_text())
    source["world_size"] = 4
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="world_size"):
        compare.load_manifest(path)
    source["world_size"] = 2
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="config_sha256"):
        compare.load_manifest(path, expect={"config_sha256": "0" * 64})
    del source["checkpoint_revision"]
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="missing"):
        compare.load_manifest(path)


def test_top_k_evidence_cannot_claim_a_full_vocabulary():
    reference = {"observation_type": "TOPK_LOGPROBS", "token_ids": [1, 2], "checkpoint_revision": "ab"}
    candidate = {"observation_type": "TOPK_LOGPROBS", "token_ids": [1, 2], "checkpoint_revision": "ab"}
    result = compare.compare_records(reference, candidate, require="FULL_VOCAB_RAW_LOGITS")
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["near_tie"] is False


def test_text_is_not_retokenized():
    with pytest.raises(ValueError, match="token ids"):
        compare.compare_records(
            {"observation_type": "TOKEN_IDS_ONLY", "text": "Paris", "checkpoint_revision": "ab"},
            {"observation_type": "TOKEN_IDS_ONLY", "text": "Paris", "checkpoint_revision": "ab"},
        )


def test_a_large_logit_error_is_not_a_near_tie():
    stability = compare.logit_stability([10.0, 0.0, 0.0], [0.0, 100.0, 0.0])
    assert stability["stability"] == "inconclusive"
    assert stability["near_tie"] is False
    assert stability["margin"] <= 2.0 * stability["epsilon"]


def test_a_wide_margin_with_a_small_error_stays_on_the_same_token():
    left = {"observation_type": "FULL_VOCAB_RAW_LOGITS", "checkpoint_revision": "ab", "logits": [5.0, 0.0, 0.0]}
    right = {"observation_type": "FULL_VOCAB_RAW_LOGITS", "checkpoint_revision": "ab", "logits": [5.1, 0.1, 0.0]}
    result = compare.compare_records(left, right)
    assert result["status"] == "PASS"
    assert result["stability"] == "conclusive"
    assert result["top_token_stable"] is True
    assert result["near_tie"] is False


def test_token_ids_match_without_inventing_probabilities():
    left = {"observation_type": "TOKEN_IDS_ONLY", "checkpoint_revision": "ab", "token_ids": [4, 5]}
    right = {"observation_type": "TOKEN_IDS_ONLY", "checkpoint_revision": "ab", "token_ids": [4, 5]}
    assert compare.compare_records(left, right)["token_parity"] == "PASS"
    right["token_ids"] = [4, 6]
    assert compare.compare_records(left, right)["status"] == "DIVERGE"


def test_package_writer_rejects_text_without_ids(tmp_path: Path):
    manifest = json.loads((NOTES / "nvidia-glm-nvfp4-manifest.json").read_text())
    with pytest.raises(ValueError, match="token_ids"):
        record.write_package(tmp_path / "pkg", manifest, [{"observation_type": "TOKEN_IDS_ONLY", "text": "Paris"}])
