"""NVFP4 startup decisions from the raw environment, before CUDA or NCCL exist."""

from __future__ import annotations

import math

_ALLOWED = ("", "0", "1", "auto")


def _positive(value: object, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} {value!r} is not a finite positive scale")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} {value!r} must be finite and positive")


def admit_recipe(block: dict) -> None:
    """Accept this export's static NVFP4 group-16 recipe. Neighbor recipes raise with the field that failed."""

    method = str(block.get("quant_method", "")).lower()
    if method == "compressed-tensors":
        raise ValueError("compressed-tensors is a later checkpoint")
    if method != "modelopt":
        raise ValueError(f"quant_method {method!r} is not modelopt")
    algo = block.get("quant_algo")
    if algo != "NVFP4":
        raise ValueError(f"quant_algo {algo!r} is not NVFP4")
    groups = block.get("config_groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise ValueError("NVFP4 recipe has no config_groups")
    for name, group in groups.items():
        if not isinstance(group, dict):
            raise ValueError(f"{name} is not a config group")
        for side in ("weights", "input_activations"):
            spec = group.get(side)
            if not isinstance(spec, dict):
                raise ValueError(f"{name} {side} are missing")
            if spec.get("dynamic") is True:
                raise ValueError(f"{name} {side} are dynamic")
            bits, kind = spec.get("num_bits"), spec.get("type")
            if bits != 4 or kind != "float":
                raise ValueError(f"{name} {side} are {bits}-bit {kind}, not 4-bit float")
            if spec.get("group_size") != 16:
                raise ValueError(f"{name} {side} group_size {spec.get('group_size')!r} is not 16")
            _positive(spec.get("global_scale"), f"{name} {side} global_scale")
    _positive(block.get("global_scale"), "global_scale")


def decide(raw_mtp: str | None, *, no_drafts: bool, drafter: bool, parallel: bool) -> str:
    """The serial NVFP4 path, or a ``ValueError`` that names the unsupported setting.

    ``raw_mtp`` is ``TF_GLM_MTP`` before the MLX/EXL3 default of ``"1"`` is applied.
    ``None`` and ``""`` are unset. Unset, ``0``, and ``auto`` with ``--no-drafts`` leave the MTP head unloaded.
    Explicit ``1``, a drafter, drafts left on, and ``--parallel`` are refused. This does not construct an engine.
    """

    if parallel:
        raise ValueError("GLM NVFP4 does not serve --parallel")
    if drafter or not no_drafts:
        raise ValueError("GLM NVFP4 drafts are unqualified: pass --no-drafts and do not pass --drafter")
    value = ("" if raw_mtp is None else raw_mtp).strip().lower()
    if value not in _ALLOWED:
        raise ValueError(f"TF_GLM_MTP: 0, 1 or auto, not {value!r}")
    if value == "1":
        raise ValueError("GLM NVFP4 MTP head is unqualified")
    return "serial"
