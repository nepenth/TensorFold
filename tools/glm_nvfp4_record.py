"""Offline records for a GLM NVFP4 reference. No model is loaded and text is never retokenized."""

from __future__ import annotations

import json
from pathlib import Path


def _compare():
    import importlib.util

    path = Path(__file__).with_name("glm_nvfp4_compare.py")
    spec = importlib.util.spec_from_file_location("glm_nvfp4_compare", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LOADED = _compare()
OBSERVATION_TYPES = _LOADED.OBSERVATION_TYPES
dump_manifest = _LOADED.dump_manifest
load_manifest = _LOADED.load_manifest

__all__ = ["OBSERVATION_TYPES", "write_package"]


def write_package(directory: str | Path, manifest: dict, observations: list[dict]) -> Path:
    """Write a package of JSON records. Tensor payloads are separate files named by the observation, never pickle."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    (root / "tensors").mkdir()
    dump_manifest(root / "manifest.json", manifest)
    load_manifest(root / "manifest.json")
    lines = []
    for record in observations:
        if record.get("observation_type") not in OBSERVATION_TYPES:
            raise ValueError(f"observation_type {record.get('observation_type')!r} is not a known record")
        if "text" in record and "token_ids" not in record:
            raise ValueError("a record with text and no token_ids would have to be retokenized; pass token_ids")
        lines.append(json.dumps(record))
    (root / "observations.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    return root
