"""NVFP4 startup decisions from the raw environment, before CUDA or NCCL exist."""

from __future__ import annotations

_ALLOWED = ("", "0", "1", "auto")


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
