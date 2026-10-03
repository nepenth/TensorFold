"""Task 10A: synthetic pointer-table projection against the same lane backend."""

# ruff: noqa: E402 -- CUDA availability MUST be checked before backend imports.

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("GLM NVFP4 dispatch needs an SM 12 GPU", allow_module_level=True)

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.families.glm5_next.cuda.nvfp4_dispatch import projection
from tensorfold.families.glm5_next.cuda.nvfp4_table import PROJECTIONS, load_synthetic_layer


def _table(n=127, k=128):
    rng = torch.Generator().manual_seed(101)
    experts = {}
    for expert in range(2):
        experts[expert] = {}
        for column, name in enumerate(PROJECTIONS):
            experts[expert][name] = {
                "weight": torch.randint(0, 256, (n, k // 2), generator=rng, dtype=torch.uint8).cuda(),
                "weight_scale": torch.full((n, k // 16), 0x38 + expert, dtype=torch.uint8, device="cuda"),
                # Exact bf16 products also equal checkpoint.alpha, avoiding a rounding-policy comparison.
                "weight_scale_2": torch.tensor(2.0 ** (column - 5), device="cuda"),
                "input_scale": torch.tensor(0.25, device="cuda"),
            }
    shared = {name: torch.zeros((64, 64), dtype=torch.bfloat16, device="cuda") for name in PROJECTIONS}
    return load_synthetic_layer(layer=3, experts=experts, shared=shared)


@pytest.fixture(scope="module")
def table():
    return _table()


def _input(rows, k):
    rng = torch.Generator().manual_seed(102)
    return torch.randn((rows, k), generator=rng).to(torch.bfloat16).cuda()


def _bits(x):
    return x.view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def _binding(rows, table, out, expert=0, column=0):
    # Exercise the C++ validation as well as the public wrapper.
    checkpoint._ext().dispatch(rows.codes, rows.scales, table.words_ptr, table.bs_ptr,
                               table.n, table.k, table.alpha, expert, column, out, 1, 16)


@pytest.mark.parametrize("m,k,sk", [(1, 128, 1), (3, 1024, 2), (17, 2048, 4),
                                   (33, 4096, 8), (65, 1024, 2), (129, 128, 1)])
@pytest.mark.parametrize("column", [0, 1, 2], ids=["gate", "up", "down"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_expert_zero_second_zero_are_lane_bitwise(m, k, sk, column, dtype):
    table = _table(k=k)
    owners = table.linears
    first, second = owners[column], owners[3 + column]
    assert not torch.equal(first.words, second.words)
    # Real CUDA addresses MUST exercise the bits above bit 31.
    assert first.words.data_ptr() > 2 ** 32 and second.words.data_ptr() > 2 ** 32
    assert first.bs.data_ptr() > 2 ** 32 and second.bs.data_ptr() > 2 ** 32
    assert int(table.words_ptr[0, column]) == first.words.data_ptr()
    assert int(table.words_ptr[1, column]) == second.words.data_ptr()
    assert qmm.split_k(first.n, k) == sk
    x = _input(m, k)
    rows = checkpoint.quant4(x, first.act)
    out = torch.full((m, first.n), float("nan"), dtype=dtype, device="cuda")
    weights_before = [(lin.words.clone(), lin.bs.clone()) for lin in (first, second)]
    references = [checkpoint.matmul(checkpoint.A4, x, lin, f32=dtype == torch.float32)
                  for lin in (first, second)]
    assert not torch.equal(_bits(references[0]), _bits(references[1]))

    # Dispatch MUST use slots even without Python access to the Fp4Linear list.
    table.linears = []
    for expert in (0, 1, 0):
        ptr = out.data_ptr()
        assert projection(rows, table, expert, column, out) is out
        assert out.data_ptr() == ptr and torch.isfinite(out).all()
        assert torch.equal(_bits(out), _bits(references[expert]))
    for lin, (words, bs) in zip((first, second), weights_before):
        assert torch.equal(lin.words, words) and torch.equal(lin.bs, bs)
    # ``owners`` keeps the non-owning views' storage live through completion.
    torch.cuda.synchronize()
    assert len(owners) == 6


@pytest.mark.parametrize("n", [127, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_caller_buffer_canary_past_write(n, dtype):
    table = _table(n=n)
    x = _input(3, 128)
    rows = checkpoint.quant4(x, 0.25)
    backing = torch.full((16 + 3 * n + 64,), -123.0, dtype=dtype, device="cuda")
    out = backing[16:16 + 3 * n].view(3, n)
    before = _bits(backing).clone()
    ptr = out.data_ptr()
    assert projection(rows, table, 0, 0, out) is out
    assert out.data_ptr() == ptr
    reference = checkpoint.matmul(checkpoint.A4, x, table.linears[0], f32=dtype == torch.float32)
    assert torch.equal(_bits(out), _bits(reference))
    assert torch.equal(_bits(backing[:16]), before[:16])
    assert torch.equal(_bits(backing[16 + 3 * n:]), before[16 + 3 * n:])


@pytest.mark.parametrize("direct", [False, True], ids=["wrapper", "binding"])
@pytest.mark.parametrize("kind", ["noncontiguous", "multirow", "dtype", "shape", "internal_overlap"])
def test_invalid_destinations_raise_without_writing(table, direct, kind):
    n, m = 127, 3
    if kind == "noncontiguous":
        backing = torch.full((m, 2 * n), -123.0, device="cuda")
        out, message = backing[:, ::2], "contiguous"
    elif kind == "multirow":
        backing = torch.full((2 * m, n), -123.0, device="cuda")
        out, message = backing[::2], "contiguous"
    elif kind == "dtype":
        backing = torch.full((m, n), -123.0, dtype=torch.float16, device="cuda")
        out, message = backing, "fp32 or bf16"
    elif kind == "shape":
        backing = torch.full((m, n + 1), -123.0, device="cuda")
        out, message = backing, "shape"
    else:
        backing = torch.full((1, n), -123.0, device="cuda")
        out, message = backing.expand(m, n), "overlap"
    before = backing.clone()
    rows = checkpoint.quant4(_input(m, 128), 0.25)
    with pytest.raises((ValueError, RuntimeError), match=message):
        if direct:
            _binding(rows, table, out)
        else:
            projection(rows, table, 0, 0, out)
    assert torch.equal(backing, before)


@pytest.mark.parametrize("kind", ["codes", "scales", "words", "bs", "table"])
def test_destination_overlapping_storage_is_refused(table, kind):
    m = 1 if kind == "bs" else 3
    rows = checkpoint.quant4(_input(m, 128), 0.25)
    if kind in ("codes", "scales"):
        # Borrow the beginning of out's storage for quantized inputs; the offsets
        # give distinct pointers while both tensors still share byte storage.
        backing = torch.ones((2048,), dtype=torch.float32, device="cuda")
        out = backing[512:512 + 3 * 127].view(3, 127)
        source = backing.view(torch.uint8)
        if kind == "codes":
            rows = checkpoint.Rows4(source[:3 * 64].view(3, 64), rows.scales)
        else:
            rows = checkpoint.Rows4(rows.codes, source[:2 * 64 * 4].view(2, 64, 4))
    elif kind == "table":
        backing = torch.ones((2048,), dtype=torch.float32, device="cuda")
        pointers = backing.view(torch.int64)[:6].view(2, 3)
        pointers.copy_(table.words_ptr)
        table = replace(table, words_ptr=pointers)
        out = backing[512:512 + 3 * 127].view(3, 127)
    else:
        source = getattr(table.linears[0], kind)
        backing = source.view(torch.float32).reshape(-1)
        out = backing[4:4 + m * 127].view(m, 127)
    before = _bits(backing).clone()
    with pytest.raises(RuntimeError, match="overlap"):
        projection(rows, table, 0, 0, out)
    assert torch.equal(_bits(backing), before)


def test_device_slots_determine_addresses_and_alpha(table):
    x = _input(3, 128)
    rows = checkpoint.quant4(x, 0.25)
    out = torch.empty((3, 127), device="cuda")
    selected = replace(table, words_ptr=table.words_ptr.clone(), bs_ptr=table.bs_ptr.clone(),
                       alpha=table.alpha.clone())
    selected.words_ptr[0, 0] = table.words_ptr[1, 1]
    selected.bs_ptr[0, 0] = table.bs_ptr[1, 1]
    selected.alpha[0, 0] = table.alpha[1, 1] * 2
    assert projection(rows, selected, 0, 0, out) is out
    reference = checkpoint.matmul(checkpoint.A4, x, replace(table.linears[4], scale=table.linears[4].scale * 2),
                                  f32=True)
    assert torch.equal(_bits(out), _bits(reference))


def test_selected_n_and_k_slots_determine_projection_shape(table):
    other = _table(n=65, k=1024)
    selected = replace(table, **{field: getattr(table, field).clone()
                                for field in ("words_ptr", "bs_ptr", "n", "k", "alpha")})
    for field in ("words_ptr", "bs_ptr", "n", "k", "alpha"):
        getattr(selected, field)[1, 2] = getattr(other, field)[0, 0]
    x = _input(3, 1024)
    out = torch.empty((3, 65), device="cuda")
    assert projection(checkpoint.quant4(x, 0.25), selected, 1, 2, out) is out
    reference = checkpoint.matmul(checkpoint.A4, x, other.linears[0], f32=True)
    assert torch.equal(_bits(out), _bits(reference))


@pytest.mark.parametrize("field,dtype", [("words_ptr", torch.int32), ("bs_ptr", torch.int32),
                                       ("n", torch.int64), ("k", torch.int64), ("alpha", torch.bfloat16)])
def test_binding_refuses_wrong_table_dtypes(table, field, dtype):
    invalid = replace(table, **{field: getattr(table, field).to(dtype)})
    out = torch.full((3, 127), -123.0, device="cuda")
    with pytest.raises(RuntimeError, match="tables"):
        _binding(checkpoint.quant4(_input(3, 128), 0.25), invalid, out)
    assert (out == -123.0).all()


@pytest.mark.parametrize("expert,column", [(-1, 0), (2, 0), (0, -1), (0, 3)])
def test_binding_refuses_invalid_slot(table, expert, column):
    out = torch.full((3, 127), -123.0, device="cuda")
    with pytest.raises(RuntimeError, match="expert id|projection column"):
        _binding(checkpoint.quant4(_input(3, 128), 0.25), table, out, expert, column)
    assert (out == -123.0).all()
