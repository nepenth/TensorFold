"""CPU splits for GLM NVFP4. The CUDA prefetch path is skipped without a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfold.families.glm5_next.cuda.split import (
    check_rank_metadata, nvfp4_column_legal, read_header, rule, split_bytes, split_file, write,
)

REV = "da920bb0b9f4a06727223a349e55468e38352348"


def test_scalars_replicate_and_are_not_ambiguous():
    name = "model.language_model.layers.0.mlp.down_proj.input_scale"
    assert rule(name) == "rep"
    assert rule(name.replace("input_scale", "weight_scale_2")) == "rep"
    raw = np.array([1.25], np.float32).view(np.uint8).copy()
    for rank in (0, 1):
        data, shape = split_bytes(raw, [], 4, rule(name), rank, dtype="F32")
        assert shape == []
        assert data.tobytes() == raw.tobytes()


def test_expert_gate_row_split_keeps_logical_k():
    assert rule("model.language_model.layers.3.mlp.experts.0.gate_proj.weight") == "row"
    weight = np.arange(2048 * 2048, dtype=np.uint8)
    scale = np.arange(2048 * 256, dtype=np.uint8)
    for rank in (0, 1):
        part, shape = split_bytes(weight, [2048, 2048], 1, "row", rank, dtype="U8")
        assert shape == [1024, 2048]
        assert part[0] == weight[rank * 1024 * 2048]
        sp, ss = split_bytes(scale, [2048, 256], 1, "row", rank, dtype="F8_E4M3")
        assert ss == [1024, 256]
        assert sp.size == 1024 * 256


def test_expert_and_dense_down_column_splits():
    down = "model.language_model.layers.3.mlp.experts.0.down_proj.weight"
    assert rule(down) == "col"
    weight = np.arange(4096 * 1024, dtype=np.uint8)
    part, shape = split_bytes(weight, [4096, 1024], 1, "col", 0, dtype="U8")
    assert shape == [4096, 512]
    assert part.size == 4096 * 512
    scale = np.arange(4096 * 128, dtype=np.uint8)
    _, ss = split_bytes(scale, [4096, 128], 1, "col", 1, dtype="F8_E4M3")
    assert ss == [4096, 64]
    dense = np.arange(4096 * 6144, dtype=np.uint8)
    _, dense_shape = split_bytes(dense, [4096, 6144], 1, "col", 0, dtype="U8")
    assert dense_shape == [4096, 3072]


def test_packed_width_96_is_rejected_and_legal_widths_pass():
    with pytest.raises(ValueError, match="multiple of 64"):
        nvfp4_column_legal([8, 96], "U8")
    with pytest.raises(ValueError, match="multiple of 64"):
        split_bytes(np.zeros(8 * 96, np.uint8), [8, 96], 1, "col", 0, dtype="U8")
    nvfp4_column_legal([4096, 1024], "U8")
    nvfp4_column_legal([4096, 128], "F8_E4M3")
    with pytest.raises(ValueError, match="legal K"):
        nvfp4_column_legal([4, 12], "F8_E4M3")


def test_k_groups_stay_in_order_across_the_split():
    width = 128
    raw = np.zeros((2, width), np.uint8)
    raw[:, :64] = 1
    raw[:, 64:] = 2
    part, shape = split_bytes(raw.reshape(-1), [2, width], 1, "col", 0, dtype="U8")
    view = part.reshape(shape)
    assert set(view[:, :64].reshape(-1).tolist()) == {1}
    assert shape[1] == 64


def test_rank_folder_records_provenance_and_rejects_a_stale_revision(tmp_path: Path):
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    raw = np.arange(4 * 128, dtype=np.uint8)
    src = tmp_path / "model-00000.safetensors"
    write(str(src), [(name, "U8", [4, 128], raw)], None)
    out = tmp_path / "rank0"
    split_file(src, out, 0, provenance={"revision": REV})
    written = next(out.glob("*.rank0.safetensors"))
    header, _ = read_header(written)
    meta = header["__metadata__"]
    assert meta["nvfp4_split"] == 1 and meta["rank"] == 0 and meta["world"] == 2
    assert meta["revision"] == REV
    info = header[name]
    assert info["shape"] == [4, 64] and info["dtype"] == "U8"
    check_rank_metadata(meta, rank=0, revision=REV)
    with pytest.raises(ValueError, match="revision"):
        check_rank_metadata(meta, rank=0, revision="0" * 40)
    with pytest.raises(ValueError, match="rank"):
        check_rank_metadata(meta, rank=1, revision=REV)
    assert check_rank_metadata(None, rank=0) is None


def test_split_device_matches_split_bytes_when_cuda_is_present():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVFP4 CUDA prefetch split needs a GPU; the CPU path is the one this run qualified")
    raw = torch.arange(4 * 128, dtype=torch.uint8, device="cuda")
    cpu = raw.cpu().numpy()
    for rank in (0, 1):
        data, shape = split_bytes(cpu, [4, 128], 1, "col", rank, dtype="U8")
        from tensorfold.families.glm5_next.cuda.split import split_device

        device, device_shape = split_device(raw, [4, 128], 1, "col", rank, dtype="U8")
        assert device_shape == shape
        assert device.cpu().numpy().tobytes() == data.tobytes()
