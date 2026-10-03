"""Header census and the per-layer host reserve. No GPU and no tensor bodies."""

from tensorfold.families.glm5_next.cuda import nvfp4_admit as admit


def _info(nbytes: int) -> dict:
    return {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}


def test_census_keeps_half_a_column_and_peaks_when_raw_reads_overlap():
    headers = {
        "lm_head.weight": _info(8),
        "model.language_model.layers.0.mlp.down_proj.weight": _info(100),
        "model.language_model.layers.1.mlp.down_proj.weight": _info(80),
        "model.language_model.layers.45.mlp.experts.0.down_proj.weight": _info(1000),
    }
    plan = admit.census(headers, layers=2)
    assert plan["keep"] == 8 + 50 + 40
    assert plan["steps"][0]["need"] == 8 + 50 + 100 + 80
    assert plan["steps"][1]["need"] == 8 + 50 + 40 + 80
    assert plan["peak"] == plan["steps"][0]["need"]
    assert plan["peak_layer"] == 0
    assert plan["unsplit_extra"] == 50 + 40
    assert plan["reserve"] == 16 << 30


def test_row_split_is_narrowed_and_a_rank_folder_is_not_halved_again():
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    headers = {name: _info(64)}
    plan = admit.census(headers, layers=1)
    assert plan["keep"] == 32
    assert plan["steps"][0]["read"] == 32
    assert plan["unsplit_extra"] == 0
    split = admit.census(headers, already_split=True, layers=1)
    assert split["keep"] == 64
    assert split["steps"][0]["read"] == 64


def test_guard_stops_before_the_layer_that_misses_the_reserve(monkeypatch):
    opened = []

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        raise AssertionError(f"reserve check opened {path}")

    monkeypatch.setattr("builtins.open", spy)
    built = []

    def build(index):
        built.append(index)

    try:
        admit.guarded([0, 1], {0: 10, 1: 100}, lambda: 40, build, reserve=16)
    except admit.NoHostReserve as exc:
        assert "cgroup cap does not contain" in str(exc)
    else:
        raise AssertionError("the second layer should have been refused")
    assert built == [0]
    assert not any("cgroup" in path for path in opened)


def test_require_reserve_uses_the_supplied_available_bytes_only():
    admit.require_reserve(200, 10, reserve=16)
    try:
        admit.require_reserve(25, 10, reserve=16)
    except admit.NoHostReserve as exc:
        assert "MemAvailable is 25" in str(exc)
    else:
        raise AssertionError("25 bytes cannot cover a 16 byte reserve")
