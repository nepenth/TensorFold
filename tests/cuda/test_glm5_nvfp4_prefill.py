"""Task 10C: synthetic multi-token lane pack, scatter, and ordered combine."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm5_next.cuda.nvfp4_prefill import (  # noqa: E402
    PrefillWorkspace,
    pack_assignments,
    routed_prefill,
)

HAS_SM12 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
cuda_only = pytest.mark.skipif(not HAS_SM12, reason="lane bitwise gate needs an SM 12 GPU")
ORDER = (7, 1, 0, 6, 2, 5, 3, 4)


def _routes(case, *, device="cpu"):
    if case == "same_eight":
        values = [ORDER] * 5
    elif case == "balanced":
        values = [[e] for e in range(8)] * 2
    elif case == "empty":
        return torch.empty((0, 8), dtype=torch.int64, device=device)
    elif case == "last_id":
        values = [[7]] * 3
    elif case == "different_rows":
        values = [ORDER[row:] + ORDER[:row] for row in range(8)]
    elif case == "sparse":
        values = [[7, 0, 7, 0, 7, 0, 7, 0], [0, 7, 0, 7, 0, 7, 0, 7]]
    elif case == "worst_tiles":
        # 128 assignments need 15 descriptors, not ceil(128 / 16) == 8.
        flat = sum(([e] * 17 for e in range(7)), []) + [7] * 9
        values = [flat[row:row + 8] for row in range(0, len(flat), 8)]
    else:
        raise AssertionError(case)
    return torch.tensor(values, dtype=torch.int64, device=device)


def _assert_pack(routes, pack, tile_size):
    flat = routes.reshape(-1).tolist()
    expected_order = sorted(range(len(flat)), key=flat.__getitem__)
    expected_counts = [flat.count(e) for e in range(8)]
    expected_offsets = [sum(expected_counts[:e]) for e in range(9)]
    assert pack.counts.tolist() == expected_counts
    assert pack.offsets.tolist() == expected_offsets
    assert pack.assignment_ids.tolist() == expected_order
    assert pack.token_ids.tolist() == [i // routes.shape[1] for i in expected_order]
    assert pack.slot_ids.tolist() == [i % routes.shape[1] for i in expected_order]
    seen = []
    for expert, start, end in pack.tile_descriptors[:pack.num_tiles].tolist():
        assert expected_counts[expert] > 0
        assert expected_offsets[expert] <= start < end <= expected_offsets[expert + 1]
        assert end - start <= tile_size
        assert all(flat[expected_order[index]] == expert for index in range(start, end))
        seen.extend(range(start, end))
    assert seen == list(range(len(flat)))
    assert pack.num_tiles == sum(math.ceil(count / tile_size) for count in expected_counts)
    assert bool((pack.tile_descriptors[pack.num_tiles:] == -1).all())


@pytest.mark.parametrize("case", ["same_eight", "balanced", "empty", "last_id", "different_rows", "sparse",
                                  "worst_tiles"])
@pytest.mark.parametrize("tile_size", [1, 4, 16])
def test_pack_counts_prefix_stability_and_tile_coverage(case, tile_size):
    routes = _routes(case)
    _assert_pack(routes, pack_assignments(routes, 8, tile_size), tile_size)


@pytest.mark.parametrize("tokens", [0, 1, 15, 16, 17, 31, 32, 33])
def test_pack_counts_around_default_tile(tokens):
    routes = torch.tensor(ORDER).expand(tokens, 8).clone()
    pack = pack_assignments(routes, 8)
    _assert_pack(routes, pack, 16)
    assert pack.counts.tolist() == [tokens] * 8


def test_worst_case_capacity_and_large_small_empty_reuse():
    workspace = PrefillWorkspace.allocate(33, 8, 128, 8, device="cpu")
    pointers = [t.data_ptr() for t in workspace._buffers()]
    for routes in (torch.tensor(ORDER).expand(33, 8).clone(), _routes("worst_tiles"),
                   _routes("sparse"), _routes("empty"), _routes("same_eight")):
        pack = pack_assignments(routes, 8, workspace=workspace)
        _assert_pack(routes, pack, 16)
        for indices in (workspace.assignment_ids, workspace.token_ids, workspace.slot_ids):
            assert bool((indices[routes.numel():] == -1).all())
        assert [t.data_ptr() for t in workspace._buffers()] == pointers
    worst = pack_assignments(_routes("worst_tiles"), 8)
    assert worst.num_tiles == worst.tile_descriptors.shape[0] == 15


@pytest.mark.parametrize("bad", [torch.tensor([[8]]), torch.tensor([[-1]]), torch.tensor([[1.5]]),
                                 torch.empty((2, 0), dtype=torch.int64), torch.tensor([1, 2])])
def test_bad_routes_are_refused_before_workspace_writes(bad):
    workspace = PrefillWorkspace.allocate(8, 8, 128, 8, device="cpu")
    before = [t.clone() for t in workspace._buffers()[:6]]
    with pytest.raises(ValueError):
        pack_assignments(bad, 8, workspace=workspace)
    assert all(torch.equal(t, old) for t, old in zip(workspace._buffers(), before))


@pytest.mark.parametrize("change", ["capacity", "slots", "experts", "tile_size"])
def test_mismatched_workspace_is_refused(change):
    workspace = PrefillWorkspace.allocate(2, 8, 128, 8, device="cpu")
    routes = _routes("sparse")
    experts, tile = 8, 16
    if change == "capacity":
        routes = _routes("same_eight")
    elif change == "slots":
        routes = _routes("last_id")
    elif change == "experts":
        experts = 9
    else:
        tile = 4
    with pytest.raises(ValueError, match="workspace"):
        pack_assignments(routes, experts, tile, workspace=workspace)


@pytest.fixture(scope="module")
def synthetic_table():
    if not HAS_SM12:
        pytest.skip("synthetic NVFP4 numeric table needs an SM 12 GPU")
    from tensorfold.cuda.nvfp4 import checkpoint
    from tensorfold.cuda.nvfp4.linear import Fp4Linear
    from tensorfold.families.glm5_next.cuda.nvfp4_table import RoutedTable, rounded_alpha

    experts = []
    for expert, code in enumerate((0x11, 0x22, 0x24, 0x26, 0x31, 0x42, 0x53, 0x64)):
        factor = (expert + 1) / 32
        triple = []
        for packed, scale, act in ((code, factor, 0.25), (code ^ 0x02, factor * 2, 0.25),
                                   (code ^ 0x04, factor / 2, 0.125)):
            triple.append(Fp4Linear.from_checkpoint(
                torch.full((128, 64), packed, dtype=torch.uint8, device="cuda"),
                torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda"), scale, act=act))
        experts.append(tuple(triple))
    owners = [lin for triple in experts for lin in triple]
    # Asymmetric numeric fixture bound directly to the same storage owners;
    # the synthetic loader's uniform-scale policy remains unchanged.
    words = torch.tensor([lin.words.data_ptr() for lin in owners], dtype=torch.int64, device="cuda").view(8, 3)
    scales = torch.tensor([lin.bs.data_ptr() for lin in owners], dtype=torch.int64, device="cuda").view(8, 3)
    ns = torch.tensor([lin.n for lin in owners], dtype=torch.int32, device="cuda").view(8, 3)
    ks = torch.tensor([lin.k for lin in owners], dtype=torch.int32, device="cuda").view(8, 3)
    factors = [rounded_alpha(lin.act, lin.scale) for lin in owners]
    assert factors == [checkpoint.alpha(lin.act, lin.scale) for lin in owners]
    alpha = torch.tensor(factors, dtype=torch.float32, device="cuda").view(8, 3)
    rng = torch.Generator().manual_seed(910)
    shared = {key: (torch.randn((128, 128), generator=rng) * 0.125).to(torch.bfloat16).cuda()
              for key in ("gate_proj", "up_proj", "down_proj")}
    table = RoutedTable(owners, words, scales, ns, ks, alpha, shared, experts=8, layer=3)
    assert len({lin[0].scale for lin in experts}) == 8
    assert len({lin[0].words.data_ptr() for lin in experts}) == 8
    yield table, tuple(experts)
    torch.cuda.synchronize()


def _inputs(routes):
    rows = routes.shape[0]
    rng = torch.Generator().manual_seed(91)
    x = torch.randn((rows, 128), generator=rng).to(torch.bfloat16).cuda()
    weights = ((torch.arange(routes.numel(), device="cuda") % 13) / 16).view_as(routes).float()
    return x, weights


def _bits(tensor):
    return tensor.view(torch.int32)


def _reference(x, table, experts, ids, weights):
    from tensorfold.families.glm5_next.cuda.nvfp4_eager import routed_eager

    out = torch.empty((x.shape[0], 128), dtype=torch.float32, device="cuda")
    if x.shape[0]:
        routed_eager(x, experts, table.shared, ids, weights, out, backend="lane")
    return out


@cuda_only
@pytest.mark.parametrize("case", ["same_eight", "balanced", "empty", "last_id", "different_rows", "sparse",
                                  "worst_tiles"])
def test_prefill_is_bitwise_with_lane_eager_and_preserves_canaries(synthetic_table, case):
    table, experts = synthetic_table
    ids = _routes(case, device="cuda")
    x, weights = _inputs(ids)
    rows = x.shape[0]
    backing = torch.full((16 + rows * 128 + 64,), -123.0, device="cuda")
    out = backing[16:16 + rows * 128].view(rows, 128)
    before = _bits(backing).clone()
    expected = _reference(x, table, experts, ids, weights)
    for _ in range(2):
        assert routed_prefill(x, table, ids, weights, out, tile_size=4) is out
        assert torch.equal(_bits(out), _bits(expected))
        assert bool(torch.isfinite(out).all())
        assert torch.equal(_bits(backing[:16]), before[:16])
        assert torch.equal(_bits(backing[16 + rows * 128:]), before[16 + rows * 128:])


@cuda_only
@pytest.mark.parametrize("tokens", [3, 4, 5])
def test_prefill_counts_around_tile_are_lane_bitwise(synthetic_table, tokens):
    table, experts = synthetic_table
    ids = torch.tensor(ORDER, device="cuda").expand(tokens, 8).clone()
    x, weights = _inputs(ids)
    out = torch.empty((tokens, 128), dtype=torch.float32, device="cuda")
    workspace = PrefillWorkspace.allocate(tokens, 8, 128, 8, device="cuda", tile_size=4)
    routed_prefill(x, table, ids, weights, out, workspace=workspace, tile_size=4)
    assert torch.equal(_bits(out), _bits(_reference(x, table, experts, ids, weights)))
    assert workspace.counts.tolist() == [tokens] * 8


@cuda_only
def test_only_selected_assignments_run_and_scatter_precedes_one_combine(synthetic_table, monkeypatch):
    from tensorfold.cuda.nvfp4 import checkpoint, linear
    from tensorfold.families.glm5_next.cuda import glue, nvfp4_eager

    table, experts = synthetic_table
    ids = _routes("sparse", device="cuda")
    x, weights = _inputs(ids)
    expected = _reference(x, table, experts, ids, weights)
    out = torch.empty_like(expected)
    workspace = PrefillWorkspace.allocate(5, 8, 128, 8, device="cuda", tile_size=4)
    workspace.assignment_outputs.fill_(-987.0)
    workspace.combine_weights.fill_(-987.0)
    assignment, combine = nvfp4_eager._assignment, glue.combine
    seen, combines = [], []
    lookup = {id(triple[0]): expert for expert, triple in enumerate(experts)}

    def record(row, gate, up, down, backend, limit):
        assert backend == "lane"
        expert = lookup[id(gate)]
        token = (row.data_ptr() - x.data_ptr()) // (128 * x.element_size())
        value = assignment(row, gate, up, down, backend, limit)
        seen.append((expert, token, value.clone()))
        return value

    def record_combine(y, wts, destination):
        assert len(seen) == ids.numel()
        assert destination is out
        assert y.shape == (2, 9, 128) and y.dtype == torch.float32
        packed = pack_assignments(ids, 8, 4)
        for (_, token, value), slot in zip(seen, packed.slot_ids.tolist()):
            assert torch.equal(_bits(y[token, slot]), _bits(value))
        assert torch.equal(wts[:, :8], weights)
        assert torch.equal(wts[:, 8], torch.ones(2, device="cuda"))
        combines.append(destination)
        return combine(y, wts, destination)

    def fail(*args, **kwargs):
        pytest.fail("prefill MUST reuse lane assignments, with no grouped experts or prompt multiply")

    monkeypatch.setattr(nvfp4_eager, "_assignment", record)
    monkeypatch.setattr(glue, "combine", record_combine)
    monkeypatch.setattr(checkpoint, "prompt", fail)
    monkeypatch.setattr(checkpoint, "_gemm", fail)
    monkeypatch.setattr(linear, "_ext", fail)  # experts.cu's extension is never called here.
    routed_prefill(x, table, ids, weights, out, workspace=workspace, tile_size=4)
    assert [expert for expert, _, _ in seen] == [0] * 8 + [7] * 8
    assert combines == [out]
    assert torch.equal(_bits(out), _bits(expected))
    assert bool((workspace.assignment_outputs[2:] == -987.0).all())
    assert bool((workspace.combine_weights[2:] == -987.0).all())


@cuda_only
def test_combine_uses_slot_order_when_expert_order_changes_rounding(synthetic_table, monkeypatch):
    from tensorfold.families.glm5_next.cuda import nvfp4_eager

    table, _ = synthetic_table
    ids = torch.tensor([[0, 2, 1, 3, 4, 5, 6, 7], [1, 0, 2, 3, 4, 5, 6, 7]], device="cuda")
    x, _ = _inputs(ids)
    weights = torch.ones((2, 8), dtype=torch.float32, device="cuda")
    out = torch.empty((2, 128), dtype=torch.float32, device="cuda")
    lookup = {id(table.linears[3 * e]): e for e in range(8)}
    values = (1e20, -1e20, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def assignment(row, gate, up, down, backend, limit):
        return torch.full((128,), values[lookup[id(gate)]], dtype=torch.float32, device=row.device)

    def shared(rows, weights, limit):
        return torch.zeros((rows.shape[0], 128), dtype=torch.float32, device=rows.device)

    # Isolate the ordering contract with nonassociative fp32 operands. The real
    # assignment's numeric bits are covered separately against the lane oracle.
    monkeypatch.setattr(nvfp4_eager, "_assignment", assignment)
    monkeypatch.setattr(nvfp4_eager, "_shared_mlp", shared)
    routed_prefill(x, table, ids, weights, out)
    assert torch.equal(_bits(out[0]), _bits(torch.zeros_like(out[0])))
    assert torch.equal(_bits(out[1]), _bits(torch.full_like(out[1], 3.0)))


@cuda_only
def test_large_then_small_then_empty_then_large_reuses_storage(synthetic_table):
    table, experts = synthetic_table
    workspace = PrefillWorkspace.allocate(17, 8, 128, 8, device="cuda", tile_size=4)
    pointers = [t.data_ptr() for t in workspace._buffers()]
    backing = torch.full((17, 128), -321.0, dtype=torch.float32, device="cuda")
    for rows in (17, 2, 0, 17):
        ids = torch.tensor(ORDER, device="cuda").expand(rows, 8).clone()
        if rows == 2:
            ids = _routes("sparse", device="cuda")
        x, weights = _inputs(ids)
        out = backing[:rows]
        tail = _bits(backing[rows:]).clone()
        slot_tail = _bits(workspace.assignment_outputs[rows:]).clone()
        routed_prefill(x, table, ids, weights, out, workspace=workspace, tile_size=4)
        assert torch.equal(_bits(out), _bits(_reference(x, table, experts, ids, weights)))
        assert torch.equal(_bits(backing[rows:]), tail)
        assert torch.equal(_bits(workspace.assignment_outputs[rows:]), slot_tail)
        pack = pack_assignments(ids, 8, 4, workspace=workspace)
        _assert_pack(ids, pack, 4)
        assert [t.data_ptr() for t in workspace._buffers()] == pointers


@cuda_only
def test_empty_batch_runs_no_numeric_work(synthetic_table, monkeypatch):
    from tensorfold.families.glm5_next.cuda import glue, nvfp4_eager

    table, _ = synthetic_table
    ids = _routes("empty", device="cuda")
    x, weights = _inputs(ids)
    out = torch.empty((0, 128), dtype=torch.float32, device="cuda")

    def fail(*args, **kwargs):
        pytest.fail("empty batch MUST launch no assignment, shared MLP, or combine")

    monkeypatch.setattr(nvfp4_eager, "_assignment", fail)
    monkeypatch.setattr(nvfp4_eager, "_shared_mlp", fail)
    monkeypatch.setattr(glue, "combine", fail)
    assert routed_prefill(x, table, ids, weights, out) is out


@cuda_only
@pytest.mark.parametrize("kind", ["noncontiguous", "wrong_dtype", "wrong_shape", "routing_overlap",
                                  "bad_id", "small_workspace"])
def test_refusals_preserve_destination_before_numeric_work(synthetic_table, monkeypatch, kind):
    from tensorfold.families.glm5_next.cuda import glue, nvfp4_eager

    table, _ = synthetic_table
    ids = _routes("same_eight", device="cuda")
    x, weights = _inputs(ids)
    out = torch.full((5, 128), -123.0, dtype=torch.float32, device="cuda")
    workspace = None
    if kind == "noncontiguous":
        out = torch.full((5, 256), -123.0, device="cuda")[:, ::2]
    elif kind == "wrong_dtype":
        out = out.to(torch.bfloat16)
    elif kind == "wrong_shape":
        out = out[:, :64].contiguous()
    elif kind == "routing_overlap":
        weights = out[:, :8]
    elif kind == "bad_id":
        ids[0, 0] = 8
    else:
        workspace = PrefillWorkspace.allocate(4, 8, 128, 8, device="cuda")
    before = out.clone()

    def fail(*args, **kwargs):
        pytest.fail("refusal MUST precede assignment, shared MLP, and combine")

    monkeypatch.setattr(nvfp4_eager, "_assignment", fail)
    monkeypatch.setattr(nvfp4_eager, "_shared_mlp", fail)
    monkeypatch.setattr(glue, "combine", fail)
    with pytest.raises(ValueError):
        routed_prefill(x, table, ids, weights, out, workspace=workspace)
    assert torch.equal(out, before)
