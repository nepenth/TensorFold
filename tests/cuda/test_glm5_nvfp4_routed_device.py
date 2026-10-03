"""Task 10B: full routed lane decode from device-table slots, with no shard pull."""

# ruff: noqa: E402 -- CUDA availability MUST be checked before backend imports.

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("GLM NVFP4 routed device decode needs an SM 12 GPU", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.linear import Fp4Linear
from tensorfold.families.glm5_next.cuda import glue, nvfp4_dispatch
from tensorfold.families.glm5_next.cuda.nvfp4_eager import routed_eager
from tensorfold.families.glm5_next.cuda.nvfp4_routed import routed_decode
from tensorfold.families.glm5_next.cuda.nvfp4_table import RoutedTable, rounded_alpha

RESIDUAL_ACT, INTERMEDIATE_ACT = 0.25, 0.125


class _NoLinears:
    def __iter__(self):
        pytest.fail("device decode MUST read the table slots, not Fp4Linear owners")

    def __getitem__(self, key):
        pytest.fail("device decode MUST read the table slots, not Fp4Linear owners")

    def __len__(self):
        pytest.fail("device decode MUST read the table slots, not Fp4Linear owners")


@pytest.fixture(scope="module")
def fixture():
    experts = []
    codes = (0x11, 0x22, 0x24, 0x26, 0x31, 0x42, 0x53, 0x64)
    for expert, code in enumerate(codes):
        factor = (expert + 1) / 32
        triple = []
        for packed, scale, act in ((code, factor, RESIDUAL_ACT),
                                   (code ^ 0x02, factor * 2, RESIDUAL_ACT),
                                   (code ^ 0x04, factor / 2, INTERMEDIATE_ACT)):
            triple.append(Fp4Linear.from_checkpoint(
                torch.full((128, 64), packed, dtype=torch.uint8, device="cuda"),
                torch.full((128, 8), 0x38, dtype=torch.uint8, device="cuda"), scale, act=act))
        experts.append(tuple(triple))
    owners = [lin for triple in experts for lin in triple]
    # Task 6's uniform-scale synthetic loader intentionally refuses this
    # asymmetric fixture. Bind owners directly without changing loader policy.
    words = torch.tensor([lin.words.data_ptr() for lin in owners], dtype=torch.int64, device="cuda").view(8, 3)
    bs = torch.tensor([lin.bs.data_ptr() for lin in owners], dtype=torch.int64, device="cuda").view(8, 3)
    ns = torch.tensor([lin.n for lin in owners], dtype=torch.int32, device="cuda").view(8, 3)
    ks = torch.tensor([lin.k for lin in owners], dtype=torch.int32, device="cuda").view(8, 3)
    alphas = [rounded_alpha(lin.act, lin.scale) for lin in owners]
    # Compare dispatch to lane with identical operands/output factors. Table's
    # bf16-up policy and checkpoint's fp32 product coincide for these factors.
    assert alphas == [checkpoint.alpha(lin.act, lin.scale) for lin in owners]
    alpha = torch.tensor(alphas, dtype=torch.float32, device="cuda").view(8, 3)
    shared = {"gate_proj": torch.full((128, 128), 0.25, dtype=torch.bfloat16, device="cuda"),
              "up_proj": torch.full((128, 128), 0.125, dtype=torch.bfloat16, device="cuda"),
              "down_proj": torch.full((128, 128), 0.5, dtype=torch.bfloat16, device="cuda")}
    table = RoutedTable(owners, words, bs, ns, ks, alpha, shared, experts=8, layer=3)
    table.linears = _NoLinears()
    yield table, tuple(experts)
    # The local experts/owners retain the non-owning pointer storage.
    torch.cuda.synchronize()


def _input():
    return torch.linspace(-0.25, 1.75, 128, device="cuda").to(torch.bfloat16).view(1, 128)


def _ids(order=(7, 1, 0, 6, 2, 5, 3, 4)):
    return torch.tensor([order], dtype=torch.int64, device="cuda")


def _weights():
    return torch.tensor([[0.5, 0.25, 0.0, 0.125, 0.375, 0.75, 0.0625, 0.875]], device="cuda")


def _out():
    return torch.full((1, 128), float("nan"), dtype=torch.float32, device="cuda")


def _bits(x):
    return x.view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def _decode(x, table, ids, weights, out, **kwargs):
    return routed_decode(x, table, ids, weights, out, residual_act=RESIDUAL_ACT,
                         intermediate_act=INTERMEDIATE_ACT, **kwargs)


def _oracle(x, table, experts, ids, weights, limit=10.0):
    return routed_eager(x, experts, table.shared, ids, weights, _out(), backend="lane", limit=limit)


def _shared_only(x, shared):
    gu = torch.cat((torch.nn.functional.linear(x, shared["gate_proj"]),
                    torch.nn.functional.linear(x, shared["up_proj"])), dim=1)
    activated = torch.empty((1, 128), dtype=torch.bfloat16, device="cuda")
    sums = torch.empty((1, 2), dtype=torch.float32, device="cuda")
    glue.swiglu(gu, activated, sums, 10.0)
    return torch.nn.functional.linear(activated.float(), shared["down_proj"].float())


def test_eight_asymmetric_experts_repeated_routes_are_lane_bitwise(fixture):
    table, experts = fixture
    assert len(experts) == 8
    assert len({triple[0].scale for triple in experts}) == 8
    assert all(not torch.equal(experts[0][0].words, triple[0].words) for triple in experts[1:])
    x, weights, out = _input(), _weights(), _out()
    for ids in (_ids(), _ids(tuple(range(8))), _ids()):
        reference = _oracle(x, table, experts, ids, weights)
        ptr = out.data_ptr()
        assert _decode(x, table, ids, weights, out) is out
        assert out.data_ptr() == ptr and torch.isfinite(out).all() and out.abs().max() > 0
        assert torch.equal(_bits(out), _bits(reference))


def test_one_residual_eight_distinct_intermediate_quants_and_ordered_combine(fixture, monkeypatch):
    table, experts = fixture
    x, ids, weights, out = _input(), _ids(), _weights(), _out()
    reference = _oracle(x, table, experts, ids, weights)
    quant, project, swiglu, combine = checkpoint.quant4, nvfp4_dispatch.projection, glue.swiglu, glue.combine
    quants, projections, activations, combines = [], [], [], []

    def record_quant(values, act, tb=0):
        result = quant(values, act, tb)
        quants.append((values, act, result))
        return result

    def record_projection(rows, bound, expert, column, destination):
        assert bound is table
        projections.append((rows, expert, column, destination))
        return project(rows, bound, expert, column, destination)

    def record_swiglu(gu, activated, sums, limit):
        swiglu(gu, activated, sums, limit)
        activations.append((gu.clone(), activated, limit))

    def record_combine(y, wts, destination):
        combines.append((y.clone(), wts.clone(), destination))
        return combine(y, wts, destination)

    def fail(*args, **kwargs):
        pytest.fail("device decode MUST use projection dispatch for every NVFP4 multiply")

    monkeypatch.setattr(checkpoint, "quant4", record_quant)
    monkeypatch.setattr(nvfp4_dispatch, "projection", record_projection)
    monkeypatch.setattr(glue, "swiglu", record_swiglu)
    monkeypatch.setattr(glue, "combine", record_combine)
    for name in ("matmul", "matmul_group", "prompt", "mlp_prompt", "_lane", "_gemm"):
        monkeypatch.setattr(checkpoint, name, fail)
    assert _decode(x, table, ids, weights, out) is out
    assert torch.equal(_bits(out), _bits(reference))
    assert len(quants) == 9 and quants[0][0] is x and quants[0][1] == RESIDUAL_ACT
    assert len(projections) == 24 and len(activations) == 9
    assert len({q[2].codes.data_ptr() for q in quants}) == 9
    assert len({q[2].scales.data_ptr() for q in quants}) == 9
    for slot, expert in enumerate(ids[0].tolist()):
        gate, up, down = projections[slot * 3:slot * 3 + 3]
        assert [(p[1], p[2]) for p in (gate, up, down)] == [(expert, 0), (expert, 1), (expert, 2)]
        assert gate[0] is quants[0][2] and up[0] is quants[0][2]
        assert quants[slot + 1][0] is activations[slot][1]
        assert quants[slot + 1][1] == INTERMEDIATE_ACT and down[0] is quants[slot + 1][2]
        assert torch.equal(_bits(down[3][0]), _bits(combines[0][0][0, slot]))
    assert all(entry[2] == 10.0 for entry in activations)
    assert len(combines) == 1 and combines[0][2] is out
    assert combines[0][0].shape == (1, 9, 128)
    assert torch.equal(combines[0][1][:, :8], weights)
    assert combines[0][1][0, 8] == 1.0


def test_saturating_gate_is_clamped_at_ten(fixture):
    table, experts = fixture
    x, ids, weights = _input(), _ids((7,)), torch.ones((1, 1), device="cuda")
    gate = checkpoint.matmul_group(x, list(experts[7][:2]))[0]
    assert gate.max() > 10
    clamped = _oracle(x, table, experts, ids, weights)
    unclamped = _oracle(x, table, experts, ids, weights, limit=10000.0)
    assert not torch.equal(_bits(clamped), _bits(unclamped))
    out = _out()
    _decode(x, table, ids, weights, out)
    assert torch.equal(_bits(out), _bits(clamped))


def test_shared_contributes_with_every_routed_weight_zero(fixture):
    table, experts = fixture
    x, ids, weights, out = _input(), _ids(), torch.zeros_like(_weights()), _out()
    _decode(x, table, ids, weights, out)
    shared = _shared_only(x, table.shared)
    assert shared.abs().max() > 0
    assert torch.equal(_bits(out), _bits(shared))
    assert torch.equal(_bits(out), _bits(_oracle(x, table, experts, ids, weights)))


def test_highest_id_and_shared_slot_are_separate(fixture, monkeypatch):
    table, experts = fixture
    x, weights = _input(), torch.ones((1, 1), device="cuda")
    high, second = _out(), _out()
    _decode(x, table, _ids((7,)), weights, high)
    _decode(x, table, _ids((1,)), weights, second)
    assert torch.equal(_bits(high), _bits(_oracle(x, table, experts, _ids((7,)), weights)))
    assert not torch.equal(_bits(high), _bits(second))

    def fail(*args, **kwargs):
        pytest.fail("shared id refusal MUST precede quantization")

    monkeypatch.setattr(checkpoint, "quant4", fail)
    before = _bits(high).clone()
    with pytest.raises(ValueError, match="shared slot"):
        _decode(x, table, _ids((8,)), weights, high)
    assert torch.equal(_bits(high), before)


def test_projection_addresses_and_alpha_are_read_from_slots(fixture):
    table, experts = fixture
    # Rebind id 1 to id 7 in all five device fields, leaving Python owners alone.
    fields = {}
    for name in ("words_ptr", "bs_ptr", "n", "k", "alpha"):
        field = getattr(table, name).clone()
        field[1] = field[7]
        fields[name] = field
    rebound = replace(table, **fields)
    x, ids, weights, out = _input(), _ids((1,)), torch.ones((1, 1), device="cuda"), _out()
    _decode(x, rebound, ids, weights, out)
    reference = _oracle(x, table, experts, _ids((7,)), weights)
    assert torch.equal(_bits(out), _bits(reference))


@pytest.mark.parametrize("kind", ["noncontiguous", "dtype", "shape", "internal_overlap",
                                  "input", "routing", "shared", "words", "bs", "table"])
def test_refused_destination_raises_without_any_work_or_write(fixture, monkeypatch, kind):
    table, experts = fixture
    x, ids, weights = _input(), _ids(), _weights()
    if kind == "noncontiguous":
        backing = torch.full((1, 256), -123.0, device="cuda")
        out, message = backing[:, ::2], "contiguous"
    elif kind == "dtype":
        backing = torch.full((1, 128), -123.0, dtype=torch.bfloat16, device="cuda")
        out, message = backing, "fp32"
    elif kind == "shape":
        backing = torch.full((1, 129), -123.0, device="cuda")
        out, message = backing, "shape"
    elif kind == "internal_overlap":
        backing = torch.full((1, 1), -123.0, device="cuda")
        out, message = backing.expand(1, 128), "overlap"
    elif kind == "input":
        backing = torch.full((1, 256), -123.0, device="cuda")
        x = backing.view(torch.bfloat16)[:, :128].contiguous()
        out, message = backing[:, :128], "overlap"
    elif kind == "routing":
        backing = torch.full((128,), -123.0, device="cuda")
        weights = backing[:8].view(1, 8)
        out, message = backing.view(1, 128), "overlap"
    elif kind == "shared":
        backing = table.shared["down_proj"]
        out, message = backing.view(torch.float32).reshape(-1)[:128].view(1, 128), "overlap"
    elif kind == "words":
        backing = experts[7][2].words
        out, message = backing.view(torch.float32).reshape(-1)[:128].view(1, 128), "overlap"
    elif kind == "bs":
        backing = experts[1][0].bs
        out, message = backing.view(torch.float32).reshape(-1)[:128].view(1, 128), "overlap"
    else:
        backing = torch.zeros(128, dtype=torch.float32, device="cuda")
        backing[:24] = table.alpha.reshape(-1)
        table = replace(table, alpha=backing[:24].view(8, 3))
        out, message = backing.view(1, 128), "overlap"
    before = backing.view(torch.uint8).clone()

    def fail(*args, **kwargs):
        pytest.fail("buffer refusal MUST precede quantization, projections, activation, shared MLP, and combine")

    monkeypatch.setattr(checkpoint, "quant4", fail)
    monkeypatch.setattr(nvfp4_dispatch, "projection", fail)
    monkeypatch.setattr(glue, "swiglu", fail)
    monkeypatch.setattr(glue, "combine", fail)
    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    with pytest.raises(ValueError, match=message):
        _decode(x, table, ids, weights, out)
    assert torch.equal(backing.view(torch.uint8), before)


def test_caller_destination_canaries_survive(fixture):
    table, experts = fixture
    backing = torch.full((16 + 128 + 64,), -123.0, dtype=torch.float32, device="cuda")
    before = _bits(backing).clone()
    out = backing[16:16 + 128].view(1, 128)
    x, ids, weights = _input(), _ids(), _weights()
    reference = _oracle(x, table, experts, ids, weights)
    assert _decode(x, table, ids, weights, out) is out
    assert torch.equal(_bits(out), _bits(reference))
    assert torch.equal(_bits(backing[:16]), before[:16])
    assert torch.equal(_bits(backing[16 + 128:]), before[16 + 128:])


@pytest.mark.parametrize("kind", ["up_shape", "down_shape", "null_address", "unaligned_address", "alpha",
                                  "shared_dtype", "batched_input", "negative_id", "weight_nan",
                                  "residual_act", "intermediate_act", "limit"])
def test_invalid_decode_contract_is_refused_before_work(fixture, monkeypatch, kind):
    table, _ = fixture
    x, ids, weights, out = _input(), _ids(), _weights(), _out()
    kwargs = {"residual_act": RESIDUAL_ACT, "intermediate_act": INTERMEDIATE_ACT, "limit": 10.0}
    if kind in ("up_shape", "down_shape"):
        k = table.k.clone()
        k[7, 1 if kind == "up_shape" else 2] += 64
        table = replace(table, k=k)
    elif kind in ("null_address", "unaligned_address"):
        ptrs = table.words_ptr.clone()
        ptrs[7, 2] = 0 if kind == "null_address" else int(ptrs[7, 2]) + 4
        table = replace(table, words_ptr=ptrs)
    elif kind == "alpha":
        alpha = table.alpha.clone()
        alpha[7, 0] = float("nan")
        table = replace(table, alpha=alpha)
    elif kind == "shared_dtype":
        shared = dict(table.shared)
        shared["gate_proj"] = shared["gate_proj"].float()
        table = replace(table, shared=shared)
    elif kind == "batched_input":
        x = x.repeat(2, 1)
    elif kind == "negative_id":
        ids[0, 0] = -1
    elif kind == "weight_nan":
        weights[0, 0] = float("nan")
    else:
        kwargs[kind] = 0.0
    before = _bits(out).clone()

    def fail(*args, **kwargs):
        pytest.fail("invalid decode contract MUST be refused before computation")

    monkeypatch.setattr(checkpoint, "quant4", fail)
    monkeypatch.setattr(nvfp4_dispatch, "projection", fail)
    monkeypatch.setattr(glue, "swiglu", fail)
    monkeypatch.setattr(glue, "combine", fail)
    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    with pytest.raises(ValueError):
        routed_decode(x, table, ids, weights, out, **kwargs)
    assert torch.equal(_bits(out), before)
