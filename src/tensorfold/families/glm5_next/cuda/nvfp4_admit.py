"""Header census and the in-process host reserve for a GLM NVFP4 load.

No tensor body is read. A cgroup cap does not contain GB10 unified allocations,
so the reserve reads the host available-byte probe and never a cgroup file.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Mapping

HOST_RESERVE = 16 << 30
_ROW = (
    r"\.mlp\.experts\.\d+\.(gate|up)_proj\.",
    r"\.mlp\.shared_experts\.(gate|up)_proj\.",
    r"\.mlp\.(gate|up)_proj\.",
    r"\.self_attn\.(q|k|v)_proj\.",
    r"\.self_attn\.(q|k|v)_conv1d\.",
    r"\.self_attn\.(f_b|g_b|b)_proj\.",
    r"\.self_attn\.(A_log|dt_bias)$",
    r"\.self_attn\.(q_b|kv_b)_proj\.",
)
_COL = (
    r"\.mlp\.experts\.\d+\.down_proj\.",
    r"\.mlp\.shared_experts\.down_proj\.",
    r"\.mlp\.down_proj\.",
    r"\.self_attn\.o_proj\.",
)
_REP = (
    r"^lm_head\.", r"embed_tokens\.", r"^model\.language_model\.norm\.weight$",
    r"_layernorm\.weight$", r"\.hc_(attn|ffn)_(fn|base|scale)$", r"\.mlp\.gate\.(weight|e_score_correction_bias)$",
    r"\.self_attn\.indexer\.", r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.self_attn\.(f_a|g_a)_proj\.",
    r"\.self_attn\.o_norm\.weight$", r"\.(eh_proj)\.", r"\.(enorm|hnorm)\.weight$", r"\.shared_head\.norm\.weight$",
)
_EXL3 = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(trellis|suh|svh|mcg)$")
_EXL3_RULES = {("gate", "trellis"): "dim1", ("gate", "suh"): "rep", ("gate", "svh"): "row",
               ("down", "trellis"): "row", ("down", "suh"): "row", ("down", "svh"): "rep"}


class NoHostReserve(RuntimeError):
    """Stop before the next layer. The host does not have the reserve."""


def _rule(name: str) -> str:
    """Same split class as ``split.rule``. Inlined so a header census does not import numpy."""

    if name.startswith("model.visual."):
        return "drop"
    if name.endswith((".input_scale", ".weight_scale_2")):
        return "rep"
    match = _EXL3.search(name)
    if match:
        proj, part = match.groups()
        return "rep" if part == "mcg" else _EXL3_RULES[("gate" if proj == "up" else proj, part)]
    hits = [kind for kind, pats in (("row", _ROW), ("col", _COL), ("rep", _REP)) if any(re.search(p, name) for p in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    return hits[0]


def _raw(info: Mapping) -> int:
    start, end = info["data_offsets"]
    return int(end) - int(start)


def _layer_index(name: str) -> int | None:
    marker = ".layers."
    at = name.find(marker)
    if at < 0:
        return None
    num = name[at + len(marker):].split(".", 1)[0]
    return int(num) if num.isdigit() else None


def _skipped(name: str) -> bool:
    if "draft_head" in name or ".mtp." in name or name.startswith("mtp."):
        return True
    if name.startswith(("model.visual", "visual.")):
        return True
    return _layer_index(name) == 45


def kept_and_read(name: str, info: Mapping, *, already_split: bool) -> tuple[int, int]:
    """Bytes this rank keeps, and bytes the current reader spans. No body is read.

    A row split is already narrowed by the reader. A column split is not: the
    span is the unsplit tensor. The kept half is the only copy a resident load
    may retain.
    """

    if name == "__metadata__" or _skipped(name):
        return 0, 0
    raw = _raw(info)
    if already_split:
        return raw, raw
    kind = _rule(name)
    if kind == "drop":
        return 0, 0
    keep = raw // 2 if kind in ("row", "col", "dim1") else raw
    read = raw // 2 if kind == "row" else raw
    return keep, read


def census(headers: Mapping[str, Mapping], *, already_split: bool = False,
           layers: int | None = None) -> dict:
    """Rank-local keep, and the running peak if the packed layer overlaps the raw reads.

    ``keep`` follows the split rule. ``lm_head`` is replicated, so one rank keeps
    the full head. ``peak`` is the largest step: tensors already kept, the packed
    current layer, and the reader's raw span of this layer and the next. Layer 45
    is neither kept nor read.
    """

    non_layer = 0
    layer_keep: dict[int, int] = defaultdict(int)
    layer_read: dict[int, int] = defaultdict(int)
    for name, info in headers.items():
        keep, read = kept_and_read(name, info, already_split=already_split)
        index = _layer_index(name)
        if index is None or _skipped(name):
            non_layer += keep
            continue
        layer_keep[index] += keep
        layer_read[index] += read
    last = layers if layers is not None else (max(layer_keep, default=-1) + 1)
    steps = []
    running = non_layer
    peak = non_layer
    peak_layer = None
    for index in range(last):
        if index == 45:
            continue
        nxt = 0 if index + 1 == 45 else layer_read.get(index + 1, 0)
        need = running + layer_keep[index] + layer_read[index] + nxt
        steps.append({"layer": index, "need": need, "keep": layer_keep[index], "read": layer_read[index]})
        if need >= peak:
            peak = need
            peak_layer = index
        running += layer_keep[index]
    keep = non_layer + sum(layer_keep[index] for index in range(last) if index != 45)
    return {
        "keep": keep,
        "peak": peak,
        "peak_layer": peak_layer,
        "steps": steps,
        "reserve": HOST_RESERVE,
        "unsplit_extra": sum(row["read"] - row["keep"] for row in steps),
    }


def require_reserve(available: int, need: int, *, reserve: int = HOST_RESERVE) -> None:
    """Raise before a layer is built. Does not consult a cgroup limit."""

    have = int(available)
    required = int(need) + int(reserve)
    if have < required:
        raise NoHostReserve(
            f"need {int(need)} bytes plus a {int(reserve)} byte host reserve "
            f"({required}); MemAvailable is {have}. "
            "A cgroup cap does not contain GB10 unified allocations."
        )


def host_available() -> int:
    """Bytes the host can still give this process. Not a cgroup cap."""

    from tensorfold.cuda.capacity import _meminfo

    memory = _meminfo()
    if not memory or "MemAvailable" not in memory:
        raise NoHostReserve("MemAvailable is not readable; refusing to guess a host reserve")
    return int(memory["MemAvailable"])


def guarded(which, steps: Mapping[int, int], available: Callable[[], int], build: Callable[[int], None],
            *, reserve: int = HOST_RESERVE) -> None:
    """Call ``build`` for each layer only after that layer's reserve check."""

    for index in which:
        require_reserve(available(), steps[index], reserve=reserve)
        build(index)


def headers_from_reader(reader) -> tuple[dict, bool]:
    """Shard or rank-folder headers only. Tensor bodies stay on disk."""

    from .split import rank_files, read_header

    headers: dict = {}
    if reader.split:
        for path in rank_files(reader.dir, reader.rank):
            header, _ = read_header(path)
            header.pop("__metadata__", None)
            headers.update(header)
        return headers, True
    seen = set()
    for filename in reader.index.values():
        path = reader.dir / filename
        if path in seen:
            continue
        seen.add(path)
        header, _ = read_header(path)
        header.pop("__metadata__", None)
        headers.update(header)
    return headers, False


def steps_for(reader, layers: int) -> dict[int, int]:
    headers, already_split = headers_from_reader(reader)
    plan = census(headers, already_split=already_split, layers=layers)
    return {row["layer"]: row["need"] for row in plan["steps"]}
