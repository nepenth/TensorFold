"""Task 10D primitive only: a CUDA skip is a limitation, NEVER a graph pass."""

# ruff: noqa: E402 -- Check CUDA capability BEFORE importing the backend.

import pytest

torch = pytest.importorskip("torch", reason="LIMITATION: PyTorch is absent; CUDA graph is unqualified")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("LIMITATION: primitive graph qualification requires an SM 12 CUDA GPU", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint, linear
from tensorfold.families.glm5_next.cuda import nvfp4_dispatch, nvfp4_routed
from tensorfold.families.glm5_next.cuda.nvfp4_graph import NVFP4PrimitiveGraph
from tensorfold.families.glm5_next.cuda.nvfp4_table import PROJECTIONS, load_synthetic_layer


def _table(k):
    rng = torch.Generator().manual_seed(104)
    experts = {e: {name: {
        "weight": torch.randint(0, 256, (127, k // 2), generator=rng, dtype=torch.uint8).cuda(),
        "weight_scale": torch.full((127, k // 16), 0x38 + e, dtype=torch.uint8, device="cuda"),
        "weight_scale_2": torch.tensor(2.0 ** (column - 5), device="cuda"),
        "input_scale": torch.tensor(0.25, device="cuda"),
    } for column, name in enumerate(PROJECTIONS)} for e in range(3)}
    shared = {name: torch.zeros((64, 64), dtype=torch.bfloat16, device="cuda") for name in PROJECTIONS}
    return load_synthetic_layer(layer=3, experts=experts, shared=shared)


def _bits(x):
    return x.view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def _fail(*args, **kwargs):
    pytest.fail("primitive launch MUST NOT read device values on the host or call an eager routed/grouped path")


@pytest.mark.parametrize("m", range(1, 7))
@pytest.mark.parametrize("k", [128, 1024], ids=["single-K", "split-K"])
@pytest.mark.parametrize("column", range(3), ids=["gate", "up", "down"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_changing_routes_and_graph_eager_graph_are_lane_bitwise(m, k, column, dtype, monkeypatch):
    table = _table(k)
    rng = torch.Generator().manual_seed(105)
    x = torch.randn((m, k), generator=rng).to(torch.bfloat16).cuda()
    rows = checkpoint.quant4(x, 0.25)
    ids_a_host, ids_b_host = [0, 2, 0, 1], [1, 0, 2, 2]
    ids_a = torch.tensor(ids_a_host, dtype=torch.int64, device="cuda")
    ids_b = torch.tensor(ids_b_host, dtype=torch.int64, device="cuda")
    fixed_ids = ids_a.clone()
    n, slots = 127, len(ids_a_host)
    backing = torch.full((16 + slots * m * n + 64,), -123.0, dtype=dtype, device="cuda")
    output = backing[16:16 + slots * m * n].view(slots, m, n)
    owners = tuple(table.linears)
    fields = tuple(getattr(table, name) for name in ("words_ptr", "bs_ptr", "n", "k", "alpha"))
    pointers = tuple(t.data_ptr() for t in (fixed_ids, output, *rows, *fields))
    fields_before = tuple(t.clone() for t in fields)
    weights_before = tuple((lin.words.clone(), lin.bs.clone()) for lin in owners)
    references = []
    for e in range(3):
        ref = torch.empty((m, n), dtype=dtype, device="cuda")
        nvfp4_dispatch.projection(rows, table, e, column, ref)
        references.append(ref)
    ref_a, ref_b = (torch.stack([references[e] for e in route]) for route in (ids_a_host, ids_b_host))
    assert not torch.equal(_bits(ref_a), _bits(ref_b))
    assert all(lin.words.data_ptr() > 2 ** 32 and lin.bs.data_ptr() > 2 ** 32 for lin in owners)

    real_launch = NVFP4PrimitiveGraph._launch
    capture_seen = []

    def guarded_launch(self):
        capture_seen.append(torch.cuda.is_current_stream_capturing())
        with monkeypatch.context() as guard:
            for name in ("item", "tolist", "cpu", "numpy", "__int__", "__float__", "__bool__"):
                guard.setattr(torch.Tensor, name, _fail)
            guard.setattr(torch.cuda, "synchronize", _fail)
            real_launch(self)

    monkeypatch.setattr(NVFP4PrimitiveGraph, "_launch", guarded_launch)
    monkeypatch.setattr(nvfp4_dispatch, "projection", _fail)
    monkeypatch.setattr(nvfp4_routed, "routed_decode", _fail)
    monkeypatch.setattr(checkpoint._ext(), "dispatch", _fail)
    monkeypatch.setattr(checkpoint, "_lane", _fail)
    monkeypatch.setattr(checkpoint, "matmul_group", _fail)
    monkeypatch.setattr(linear, "_ext", _fail)  # The grouped experts extension is forbidden.
    graph = NVFP4PrimitiveGraph(rows, table, column, fixed_ids, output)
    assert capture_seen == [False, True]
    assert graph._sk == (1 if k == 128 else 2)
    # The graph retains packed owners even if Python's owner list is cleared.
    table.linears.clear()

    # A -> B -> A, then graph -> eager -> graph. Poison EVERY destination launch.
    for mode, ids, reference in (("replay", ids_a, ref_a), ("replay", ids_b, ref_b),
                                 ("replay", ids_a, ref_a), ("eager", ids_b, ref_b),
                                 ("replay", ids_a, ref_a)):
        output.fill_(float("nan"))
        graph._bank.fill_(float("nan"))
        assert getattr(graph, mode)(ids) is output
        assert torch.isfinite(output).all()
        assert torch.equal(_bits(output), _bits(reference))
        assert torch.equal(fixed_ids, ids)
        assert tuple(t.data_ptr() for t in (fixed_ids, output, *rows, *fields)) == pointers
        assert (backing[:16] == -123.0).all() and (backing[16 + slots * m * n:] == -123.0).all()
    assert capture_seen == [False, True, False]  # Replays NEVER recapture or execute Python lanes.
    for field, before in zip(fields, fields_before):
        assert torch.equal(field, before)
    for lin, (words, scales) in zip(owners, weights_before):
        assert torch.equal(lin.words, words) and torch.equal(lin.bs, scales)


def test_table_slots_are_bound_before_capture_and_replacement_is_refused():
    table = _table(128)
    # Swap only device slots, keeping owner order fixed, and change alpha.
    for name in ("words_ptr", "bs_ptr"):
        field = getattr(table, name)
        field[0, 0].copy_(field[2, 0])
    table.alpha[0, 0].mul_(2)
    x = torch.ones((1, 128), dtype=torch.bfloat16, device="cuda")
    rows = checkpoint.quant4(x, 0.25)
    ids = torch.tensor([0], dtype=torch.int64, device="cuda")
    output = torch.empty((1, 1, 127), device="cuda")
    reference = torch.empty((1, 127), device="cuda")
    nvfp4_dispatch.projection(rows, table, 0, 0, reference)
    graph = NVFP4PrimitiveGraph(rows, table, 0, ids.clone(), output)
    output.fill_(float("nan"))
    graph.replay(ids)
    assert torch.equal(_bits(output[0]), _bits(reference))
    table.words_ptr = table.words_ptr.clone()
    output.fill_(-123.0)
    with pytest.raises(RuntimeError, match="addresses or metadata changed"):
        graph.replay(ids)
    assert (output == -123.0).all()


@pytest.mark.parametrize("kind", ["host_ids", "ids_dtype", "ids_shape", "output_overlap", "prefill"])
def test_invalid_buffers_are_refused(kind):
    table = _table(128)
    x = torch.ones((7 if kind == "prefill" else 1, 128), dtype=torch.bfloat16, device="cuda")
    rows = checkpoint.quant4(x, 0.25)
    ids = torch.tensor([0], dtype=torch.int64, device="cuda")
    output = torch.full((1, x.shape[0], 127), -123.0, device="cuda")
    if kind == "host_ids":
        ids = ids.cpu()
    elif kind == "ids_dtype":
        ids = ids.int()
    elif kind == "ids_shape":
        ids = ids.view(1, 1)
    elif kind == "output_overlap":
        # Disjoint views still share destination storage and MUST be refused.
        storage = torch.ones((2048,), device="cuda")
        codes = storage.view(torch.uint8)[:64].view(1, 64)
        rows = checkpoint.Rows4(codes, rows.scales)
        output = storage[512:639].view(1, 1, 127)
    before = output.clone()
    with pytest.raises(ValueError):
        NVFP4PrimitiveGraph(rows, table, 0, ids, output)
    assert torch.equal(output, before)


def test_replay_refuses_host_ids_before_writing():
    table = _table(128)
    rows = checkpoint.quant4(torch.ones((1, 128), dtype=torch.bfloat16, device="cuda"), 0.25)
    ids = torch.zeros((1,), dtype=torch.int64, device="cuda")
    output = torch.empty((1, 1, 127), device="cuda")
    graph = NVFP4PrimitiveGraph(rows, table, 0, ids, output)
    output.fill_(-123.0)
    with pytest.raises(ValueError, match="captured CUDA device"):
        graph.replay(torch.zeros((1,), dtype=torch.int64))
    assert (output == -123.0).all()
