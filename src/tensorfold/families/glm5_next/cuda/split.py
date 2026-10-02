"""Read each rank directly from checkpoint bytes or a saved rank folder, preserving packed groups at every split boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

ROW = (
    r"\.mlp\.experts\.\d+\.(gate|up)_proj\.",
    r"\.mlp\.shared_experts\.(gate|up)_proj\.",
    r"\.mlp\.(gate|up)_proj\.",
    r"\.self_attn\.(q|k|v)_proj\.",
    r"\.self_attn\.(q|k|v)_conv1d\.",
    r"\.self_attn\.(f_b|g_b|b)_proj\.",
    r"\.self_attn\.(A_log|dt_bias)$",
    r"\.self_attn\.(q_b|kv_b)_proj\.",
)
COL = (
    r"\.mlp\.experts\.\d+\.down_proj\.",
    r"\.mlp\.shared_experts\.down_proj\.",
    r"\.mlp\.down_proj\.",
    r"\.self_attn\.o_proj\.",
)
REP = (
    r"^lm_head\.", r"embed_tokens\.", r"^model\.language_model\.norm\.weight$",
    r"_layernorm\.weight$", r"\.hc_(attn|ffn)_(fn|base|scale)$", r"\.mlp\.gate\.(weight|e_score_correction_bias)$",
    r"\.self_attn\.indexer\.", r"\.self_attn\.(q_a_proj|kv_a_proj_with_mqa)\.", r"\.self_attn\.(f_a|g_a)_proj\.",
    r"\.self_attn\.o_norm\.weight$", r"\.(eh_proj)\.", r"\.(enorm|hnorm)\.weight$", r"\.shared_head\.norm\.weight$",
)
RUN = 128 << 20          # most bytes one read of neighbouring tensors takes (``RankReader.prefetch``)
GAP = 1 << 20            # most unused bytes such a read spans between two tensors (more reads other layers twice)
READERS = 8              # reads in flight: past this the layers are built slower than they are read
DTYPE_BYTES = {"U32": 4, "I32": 4, "F32": 4, "BF16": 2, "F16": 2, "I16": 2, "U16": 2, "U8": 1, "I8": 1, "I64": 8,
               "F64": 8, "F8_E4M3": 1}
# the files a rank folder needs besides its weights (the tokenizer, chat template and configs)
SMALL = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
         "processor_config.json", "model.safetensors.index.json")


# EXL3 gate/up split tile columns and svh; down splits tile rows and suh; each replicates the remaining tensors.
EXL3_EXPERT = re.compile(r"\.mlp\.experts\.\d+\.(gate|up|down)_proj\.(trellis|suh|svh|mcg)$")
EXL3_RULES = {("gate", "trellis"): "dim1", ("gate", "suh"): "rep", ("gate", "svh"): "row",
              ("down", "trellis"): "row", ("down", "suh"): "row", ("down", "svh"): "rep"}


def rule(name: str) -> str:
    if name.startswith("model.visual."):
        return "drop"
    # Before the row/column scan: those patterns also match gate_proj.input_scale and would be ambiguous.
    if name.endswith((".input_scale", ".weight_scale_2")):
        return "rep"
    m = EXL3_EXPERT.search(name)
    if m:
        proj, part = m.groups()
        return "rep" if part == "mcg" else EXL3_RULES[("gate" if proj == "up" else proj, part)]
    hits = [kind for kind, pats in (("row", ROW), ("col", COL), ("rep", REP)) if any(re.search(p, name) for p in pats)]
    if len(hits) != 1:
        raise ValueError(f"{name}: split rule is ambiguous or missing ({hits})")
    return hits[0]


def read_header(path: str | Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def nvfp4_column_legal(shape: list[int], dtype: str) -> None:
    """TP=2 column split: each rank's logical K must be a multiple of 64.

    For a packed ``U8`` weight, stored columns ``P`` are ``K/2`` and each rank's logical K is ``P``, so ``P % 64 == 0``
    on the original tensor. ``P % 32 == 0`` lets ``P = 96`` through. For an ``F8_E4M3`` scale, stored columns are
    ``K/16`` and each rank's logical K is ``8 * columns``.
    """

    if dtype == "U8":
        if len(shape) < 2 or shape[1] % 64:
            raise ValueError(f"NVFP4 packed width {shape[1] if len(shape) > 1 else shape} is not a multiple of 64")
    elif dtype == "F8_E4M3":
        if len(shape) < 2 or shape[1] % 8:
            raise ValueError(f"NVFP4 scale width {shape[1] if len(shape) > 1 else shape} does not leave a legal K")


def split_bytes(raw: np.ndarray, shape: list[int], itemsize: int, kind: str, rank: int, *,
                dtype: str | None = None) -> tuple[np.ndarray, list[int]]:
    """A tensor's bytes -> rank's part of them and its shape."""

    if kind == "rep":
        return raw, list(shape)
    if kind == "row":
        rows = shape[0]
        if rows % 2:
            raise ValueError(f"row split of odd leading dim {shape}")
        per = raw.size // rows
        half = rows // 2
        return raw[rank * half * per:(rank + 1) * half * per], [half] + list(shape[1:])
    if kind == "col":
        if dtype in ("U8", "F8_E4M3"):
            nvfp4_column_legal(list(shape), dtype)
        if len(shape) != 2 or shape[1] % 2:
            raise ValueError(f"column split needs an even 2-D shape, got {shape}")
        view = raw.reshape(shape[0], shape[1] * itemsize)
        half = shape[1] // 2
        part = np.ascontiguousarray(view[:, rank * half * itemsize:(rank + 1) * half * itemsize])
        return part.reshape(-1), [shape[0], half]
    if kind == "dim1":                                   # the second axis of a 2-D or higher tensor
        if len(shape) < 2 or shape[1] % 2:
            raise ValueError(f"split of the second axis needs an even second dim, got {shape}")
        inner = int(np.prod(shape[2:])) * itemsize
        view = raw.reshape(shape[0], shape[1] * inner)
        half = shape[1] // 2
        part = np.ascontiguousarray(view[:, rank * half * inner:(rank + 1) * half * inner])
        return part.reshape(-1), [shape[0], half] + list(shape[2:])
    raise ValueError(kind)


def split_device(raw, shape: list[int], itemsize: int, kind: str, rank: int, *, dtype: str | None = None):
    """``split_bytes`` for a uint8 tensor on the GPU: the rank's part (a new contiguous tensor) and its shape."""

    if kind == "rep":
        return raw.clone(), list(shape)
    if kind == "row":
        if shape[0] % 2:
            raise ValueError(f"row split of odd leading dim {shape}")
        per, half = raw.numel() // shape[0], shape[0] // 2
        return raw[rank * half * per:(rank + 1) * half * per].clone(), [half] + list(shape[1:])
    if kind in ("col", "dim1"):
        if kind == "col" and dtype in ("U8", "F8_E4M3"):
            nvfp4_column_legal(list(shape), dtype)
        if len(shape) < 2 or shape[1] % 2 or (kind == "col" and len(shape) != 2):
            raise ValueError(f"{kind} split needs an even second dim, got {shape}")
        inner = int(np.prod(shape[2:])) * itemsize
        half = shape[1] // 2
        part = raw.view(shape[0], shape[1] * inner)[:, rank * half * inner:(rank + 1) * half * inner]
        return part.contiguous().reshape(-1), [shape[0], half] + list(shape[2:])
    raise ValueError(kind)


def torch_dtype(dtype: str):
    import torch

    return {"U32": torch.uint32, "I32": torch.int32, "F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16,
            "I16": torch.int16, "U16": torch.uint16, "U8": torch.uint8, "I8": torch.int8, "I64": torch.int64,
            "F64": torch.float64, "F8_E4M3": torch.float8_e4m3fn}[dtype]


def rank_files(model_dir: str | Path, rank: int) -> list[Path]:
    return sorted(Path(model_dir).glob(f"*.rank{rank}.safetensors"))


class RankReader:
    """Read stored-dtype CPU tensors for one rank from the full checkpoint or its pre-split folder."""

    def __init__(self, model_dir: str | Path, rank: int) -> None:
        from tensorfold.cuda.direct_read import ReadAhead, Reader, SafeTensors

        self.dir, self.rank = Path(model_dir), rank
        self.io = Reader()                                # O_DIRECT reads where the file system allows them
        self.reads = ReadAhead(self.io, READERS, RUN, GAP)
        mine, other = rank_files(self.dir, rank), rank_files(self.dir, 1 - rank)
        if other and not mine:
            raise ValueError(f"{self.dir} holds rank {1 - rank}'s share: give rank {rank} its own folder or the "
                             "full checkpoint")
        self.split = bool(mine)
        if self.split:
            revision_file = self.dir / "nvfp4-revision.txt"
            expected = revision_file.read_text().strip() if revision_file.is_file() else None
            for path in mine:
                header, _ = read_header(path)
                check_rank_metadata(header.get("__metadata__"), rank=rank, revision=expected)
            self.folder = SafeTensors(mine, self.io)
            self.index = dict.fromkeys(self.folder.keys())
            return
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            names = json.loads(index.read_text())["weight_map"]
        else:
            names = {k: p.name for p in sorted(self.dir.glob("*.safetensors")) for k in read_header(p)[0]
                     if k != "__metadata__"}
        self.files: dict[str, tuple[dict, int]] = {}
        self.index = dict(names)

    def prefetch(self, names, device=None) -> None:
        """Start reading ``names`` (a layer's hundreds of small expert tensors) in shared reads; with a CUDA ``device``, ``get`` returns them uploaded."""

        items = []
        for name in dict.fromkeys(names):
            span = self._span(name)
            items.append((name, span[0], span[1], span[2], span))
        self.reads.queue(items, device, self._cut)

    @property
    def ahead(self) -> dict:
        return self.reads.ahead

    def get(self, name: str):
        out = self.reads.take(name)
        return out if out is not None else self._read(name)

    def close(self) -> None:
        self.reads.close()

    def _span(self, name: str) -> tuple[str, int, int, str, list[int], str]:
        """(file, first byte, end byte, split kind, shape, dtype) of the bytes this rank reads for ``name``."""

        if self.split:                                    # a rank folder holds the rank's tensors as they are
            path, begin, n, dtype, shape = self.folder.where[name]
            return str(path), begin, begin + n, "rep", list(shape), dtype
        file = str(self.dir / self.index[name])
        if file not in self.files:
            self.files[file] = read_header(file)
        header, base = self.files[file]
        info = header[name]
        kind = rule(name)
        if kind == "drop":
            raise KeyError(f"{name} is not used by the engine")
        a, b = info["data_offsets"]
        shape = list(info["shape"])
        if kind == "row" and shape and shape[0] % 2 == 0:   # the rank's rows are one run: read only those
            per = (b - a) // shape[0] * (shape[0] // 2)
            a, b, kind, shape = a + self.rank * per, a + (self.rank + 1) * per, "rep", [shape[0] // 2] + shape[1:]
        return file, base + a, base + b, kind, shape, info["dtype"]

    def _tensor(self, raw: np.ndarray, span: tuple, own: bool):
        """The rank's tensor from the span's bytes; ``own``: never a view of ``raw`` (a shared read's buffer)."""

        import torch

        _, _, _, kind, shape, dtype = span
        data, shape = split_bytes(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, dtype=dtype)
        if own and np.may_share_memory(data, raw):
            data = data.copy()
        return torch.from_numpy(data).view(torch_dtype(dtype)).reshape(shape)

    def _read(self, name: str):
        if self.split:
            return self.folder.get(name)
        span = self._span(name)
        return self._tensor(self.io.read(span[0], span[1], span[2] - span[1]).numpy(), span, own=False)

    def _cut(self, raw, span: tuple):
        """The rank's tensor from its span's bytes, a view of a shared read (so copied), on the host or the device."""

        if not raw.is_cuda:
            return self._tensor(raw.numpy(), span, own=True)
        _, _, _, kind, shape, dtype = span
        data, shape = split_device(raw, shape, DTYPE_BYTES[dtype], kind, self.rank, dtype=dtype)
        return data.view(torch_dtype(dtype)).reshape(shape)

def write(path: str, tensors: list[tuple[str, str, list[int], np.ndarray]], metadata: dict | None) -> None:
    header: dict = {}
    offset = 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.size]}
        offset += data.size
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in tensors:
            f.write(memoryview(data))
    os.replace(tmp, path)


def check_rank_metadata(metadata: dict | None, *, rank: int, revision: str | None = None) -> None:
    """Reject an NVFP4 rank folder whose marker names the wrong rank, world, or source revision.

    Folders without ``nvfp4_split`` are the MLX and EXL3 rank folders and are left alone.
    """

    if not metadata or "nvfp4_split" not in metadata:
        return
    if metadata.get("nvfp4_split") != 1:
        raise ValueError(f"NVFP4 split format {metadata.get('nvfp4_split')!r} is not 1")
    if int(metadata.get("world", -1)) != 2:
        raise ValueError(f"NVFP4 rank folder world {metadata.get('world')!r} is not 2")
    if int(metadata.get("rank", -1)) != rank:
        raise ValueError(f"NVFP4 rank folder is rank {metadata.get('rank')!r}, not {rank}")
    if revision is not None and metadata.get("revision") != revision:
        raise ValueError(f"NVFP4 rank folder revision {metadata.get('revision')!r} is not {revision}")


def split_file(src: str | Path, out: str | Path, rank: int, *, provenance: dict | None = None) -> dict:
    """One checkpoint file -> OUT/<stem>.rank<R>.safetensors with the rank's part of every tensor it keeps."""

    header, base = read_header(src)
    metadata = header.pop("__metadata__", None)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    stem = os.path.basename(str(src)).replace(".safetensors", "")
    summary = {"rep": 0, "row": 0, "col": 0, "dim1": 0, "drop": 0}
    part = []
    for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
        info = header[name]
        kind = rule(name)
        summary[kind] += 1
        if kind == "drop":
            continue
        a, b = info["data_offsets"]
        itemsize = DTYPE_BYTES[info["dtype"]]
        data, shape = split_bytes(mm[base + a:base + b], info["shape"], itemsize, kind, rank, dtype=info["dtype"])
        if int(np.prod(shape)) * itemsize != data.size:
            raise ValueError(f"{name}: {shape} does not match {data.size} bytes")
        part.append((name, info["dtype"], shape, data))
    os.makedirs(out, exist_ok=True)
    if provenance:
        metadata = dict(metadata or {})
        metadata.update(provenance)
        metadata["nvfp4_split"] = 1
        metadata["world"] = 2
        metadata["rank"] = rank
    if part:
        write(os.path.join(str(out), f"{stem}.rank{rank}.safetensors"), part, metadata)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="the checkpoint (e.g. the snapshot `tensorfold pull` downloaded)")
    p.add_argument("--rank", type=int, choices=(0, 1), required=True, help="the rank this machine serves")
    p.add_argument("out", type=Path, help="the folder to write (then: tensorfold serve OUT --tp 2 --rank R ...)")
    args = p.parse_args(argv)
    files = sorted(args.model_dir.glob("model-*.safetensors"))
    if not files:
        raise SystemExit(f"{args.model_dir}: no model-*.safetensors files")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        if (args.model_dir / name).exists():
            shutil.copyfile(args.model_dir / name, args.out / name)
    for src in files:
        stem = src.name.replace(".safetensors", "")
        if (args.out / f"{stem}.rank{args.rank}.safetensors").exists():
            continue
        print(src.name, split_file(src, args.out, args.rank), flush=True)
    print(f"rank {args.rank}'s share of {len(files)} files in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
