"""Compare saved GLM NVFP4 observations. Missing evidence stays missing."""

from __future__ import annotations

import json
import math
from pathlib import Path

OBSERVATION_TYPES = (
    "TOKEN_IDS_ONLY",
    "TOPK_LOGPROBS",
    "FULL_VOCAB_LOGPROBS",
    "FULL_VOCAB_RAW_LOGITS",
    "INTERNAL_LAYER_TRACE",
)
FULL_VOCAB = frozenset({"FULL_VOCAB_LOGPROBS", "FULL_VOCAB_RAW_LOGITS"})
BASE = "56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21"
NUMERIC_POLICY_ID = "glm-nvfp4-tensorfold-1"
REQUIRED = (
    "schema_version",
    "implementation_base",
    "implementation_head",
    "checkpoint_revision",
    "config_sha256",
    "index_sha256",
    "world_size",
    "reference_status",
    "numeric_policy_id",
)


def _hex(value: object, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} is not a {length}-character hex digest")
    return value


def load_manifest(path: str | Path, *, expect: dict | None = None) -> dict:
    """Read a manifest. Reject a missing identity, the wrong world size, a bad hash, or a foreign numeric policy."""

    data = json.loads(Path(path).read_text())
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise ValueError(f"manifest missing {', '.join(missing)}")
    if data["schema_version"] != 1:
        raise ValueError(f"manifest schema_version {data['schema_version']!r}")
    if data["implementation_base"] != BASE:
        raise ValueError(f"manifest base {data['implementation_base']!r} is not {BASE}")
    _hex(data["implementation_head"], "implementation_head", 40)
    _hex(data["checkpoint_revision"], "checkpoint_revision", 40)
    _hex(data["config_sha256"], "config_sha256", 64)
    _hex(data["index_sha256"], "index_sha256", 64)
    if data["world_size"] != 2:
        raise ValueError(f"GLM NVFP4 world_size is 2, not {data['world_size']!r}")
    if data["numeric_policy_id"] != NUMERIC_POLICY_ID:
        raise ValueError(f"numeric policy {data['numeric_policy_id']!r} is not {NUMERIC_POLICY_ID}")
    if data["reference_status"] not in ("NOT_INSPECTED", "BLOCKED_REFERENCE", "INSPECTED"):
        raise ValueError(f"reference_status {data['reference_status']!r}")
    if expect:
        for key, value in expect.items():
            if data.get(key) != value:
                raise ValueError(f"manifest {key} {data.get(key)!r} does not match {value!r}")
    return data


def dump_manifest(path: str | Path, data: dict) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
    load_manifest(path)


def _ids(record: dict) -> list[int]:
    if "token_ids" not in record:
        raise ValueError("token ids are an explicit field; text is not retokenized")
    ids = record["token_ids"]
    if not isinstance(ids, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise ValueError("token_ids must be a list of ints")
    return ids


def _finite(values: list[float], name: str) -> list[float]:
    out = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} contains a nonfinite value")
        out.append(number)
    return out


def logit_stability(reference: list[float], candidate: list[float]) -> dict:
    """Full-vocabulary stability. ``m > 2ε`` is conclusive. A smaller margin is inconclusive, not a near-tie."""

    if len(reference) < 2 or len(reference) != len(candidate):
        raise ValueError(f"vocabulary length {len(reference)} does not match {len(candidate)}")
    left, right = _finite(reference, "reference"), _finite(candidate, "candidate")
    epsilon = max(abs(a - b) for a, b in zip(left, right))
    order = sorted(range(len(left)), key=lambda i: (left[i], -i), reverse=True)
    margin = left[order[0]] - left[order[1]]
    candidate_top = max(range(len(right)), key=lambda i: (right[i], -i))
    conclusive = margin > 2.0 * epsilon
    return {
        "epsilon": epsilon,
        "margin": margin,
        "stability": "conclusive" if conclusive else "inconclusive",
        "near_tie": False,
        "top_token_stable": (candidate_top == order[0]) if conclusive else None,
    }


def compare_records(reference: dict, candidate: dict, *, require: str | None = None) -> dict:
    """One pair of observations. Top-k evidence cannot satisfy a full-vocabulary requirement."""

    for record in (reference, candidate):
        kind = record.get("observation_type")
        if kind not in OBSERVATION_TYPES:
            raise ValueError(f"observation_type {kind!r} is not a known record")
        if "text" in record and "token_ids" not in record:
            raise ValueError("token ids are an explicit field; text is not retokenized")
    if reference.get("checkpoint_revision") != candidate.get("checkpoint_revision"):
        raise ValueError("checkpoint revisions differ")
    needed = require or reference["observation_type"]
    if needed in FULL_VOCAB and reference["observation_type"] not in FULL_VOCAB:
        return {"status": "INSUFFICIENT_EVIDENCE", "missing_evidence": [needed], "token_parity": "NOT_EVALUATED",
                "near_tie": False}
    result: dict = {"status": "NOT_EVALUATED", "near_tie": False, "missing_evidence": []}
    if "token_ids" in reference or "token_ids" in candidate:
        same = _ids(reference) == _ids(candidate)
        result["token_parity"] = "PASS" if same else "DIVERGE"
        result["status"] = "PASS" if same else "DIVERGE"
    if needed in FULL_VOCAB:
        key = "logits" if needed == "FULL_VOCAB_RAW_LOGITS" else "logprobs"
        if key not in reference or key not in candidate:
            return {"status": "INSUFFICIENT_EVIDENCE", "missing_evidence": [key], "token_parity": result.get("token_parity", "NOT_EVALUATED"),
                    "near_tie": False}
        stability = logit_stability(reference[key], candidate[key])
        result.update(stability)
        if stability["stability"] == "conclusive" and stability["top_token_stable"] is False:
            result["status"] = "DIVERGE"
        elif stability["stability"] == "inconclusive":
            result["status"] = "INCONCLUSIVE"
        elif result.get("status") != "DIVERGE":
            result["status"] = "PASS"
    return result
