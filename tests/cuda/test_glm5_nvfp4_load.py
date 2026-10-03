"""Tiny local checkpoint fixtures for the GLM CUDA NVFP4 loader."""

# ruff: noqa: E402 -- Check CUDA before importing backend modules.

from __future__ import annotations

import gc
import json
import math
import weakref

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("GLM NVFP4 load fixtures need an SM 12 GPU", allow_module_level=True)

from test_glm5_nvfp4_config import _config
from tensorfold.cuda.nvfp4 import checkpoint, linear as fp4
from tensorfold.families.glm5_next.cuda import nvfp4_dispatch, nvfp4_table as table, weights
from tensorfold.families.glm5_next.cuda.qmm import B16
from tensorfold.families.glm5_next.cuda.split import RankReader, split_file, write

PREFIX = weights.PREFIX
FACTORS = {"gate_proj": (0.125, 0.5), "up_proj": (0.25, 0.5), "down_proj": (0.0625, 0.25)}


def _save(path, tensors):
    dtypes = {torch.bfloat16: "BF16", torch.float32: "F32", torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3"}
    write(str(path), [
        (name, dtypes[t.dtype], list(t.shape), t.reshape(-1).view(torch.uint8).numpy())
        for name, t in tensors.items()
    ], None)


def _checkpoint(tmp_path, *, extra=None, dense_nvfp4=False):
    """E=4, D=128, rank-local MoE width=64; BF16 KDA/dense and DSA/MoE layers."""
    config = _config()
    config["text_config"].update(
        hidden_size=128, num_hidden_layers=3, num_attention_heads=2, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=128, v_head_dim=384, moe_intermediate_size=128, intermediate_size=128,
        linear_num_heads=2, linear_conv_kernel_dim=2, hc_mult=1,
        index_n_heads=2, index_head_dim=32,
        layer_types=["linear_attention", "deepseek_sparse_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors = {}

    def bf(name, shape, dtype=torch.bfloat16):
        tensors[name] = ((torch.arange(math.prod(shape)).reshape(shape) % 31).float() / 128).to(dtype)

    bf(PREFIX + "embed_tokens.weight", (128, 128))
    bf(PREFIX + "norm.weight", (128,))
    bf("lm_head.weight", (128, 128))
    for i in range(3):
        base = PREFIX + f"layers.{i}."
        for site in ("attn", "ffn"):
            bf(base + f"hc_{site}_fn", (3, 128))
            bf(base + f"hc_{site}_base", (3,), torch.float32)
            bf(base + f"hc_{site}_scale", (3,), torch.float32)
        for name in ("input_layernorm", "post_attention_layernorm"):
            bf(base + name + ".weight", (128,))
        p = base + "self_attn."
        if i == 0:
            for proj in ("q", "k", "v", "b", "f_b", "g_b"):
                bf(p + proj + "_proj.weight", (256, 128))
            for proj in ("f_a", "g_a"):
                bf(p + proj + "_proj.weight", (128, 128))
            for proj in "qkv":
                bf(p + proj + "_conv1d.weight", (256, 1, 2))
            bf(p + "A_log", (2,), torch.float32)
            bf(p + "dt_bias", (256,), torch.float32)
            bf(p + "o_norm.weight", (128,))
            bf(p + "o_proj.weight", (128, 256))
        else:
            for proj, shape in {
                "q_a_proj": (64, 128), "kv_a_proj_with_mqa": (64, 128), "q_b_proj": (256, 64),
                "kv_b_proj": (1024, 64), "o_proj": (128, 768),
                "indexer.wk": (64, 128), "indexer.weights_proj": (2, 128), "indexer.wq_b": (64, 64),
            }.items():
                bf(p + proj + ".weight", shape)
            for name in ("q_a_layernorm", "kv_a_layernorm"):
                bf(p + name + ".weight", (64,))
            bf(p + "indexer.k_norm.weight", (32,))
            bf(p + "indexer.k_norm.bias", (32,))
            bf(p + "indexer.index_kpool_compress_gate", (4, 128))
            bf(p + "indexer.index_kpool_compress_ape", (4, 128))
        p = base + "mlp."
        if i == 0:
            for proj, (factor, act) in FACTORS.items():
                if dense_nvfp4:
                    name = p + proj + "."
                    tensors[name + "weight"] = torch.zeros((128, 64), dtype=torch.uint8)
                    tensors[name + "weight_scale"] = torch.ones((128, 8), dtype=torch.float8_e4m3fn)
                    tensors[name + "weight_scale_2"] = torch.tensor([factor])
                    tensors[name + "input_scale"] = torch.tensor([act])
                else:
                    bf(p + proj + ".weight", (128, 128))
            continue
        bf(p + "gate.weight", (4, 128))
        bf(p + "gate.e_score_correction_bias", (4,), torch.float32)
        for proj, (factor, act) in FACTORS.items():
            bf(p + f"shared_experts.{proj}.weight", (128, 128))
            for e in range(4):
                name = p + f"experts.{e}.{proj}."
                codes = (torch.arange(128 * 64).reshape(128, 64) + e * 17 + i * 3) % 256
                tensors[name + "weight"] = codes.to(torch.uint8)
                tensors[name + "weight_scale"] = torch.ones((128, 8), dtype=torch.float8_e4m3fn)
                tensors[name + "weight_scale_2"] = torch.tensor([factor])
                tensors[name + "input_scale"] = torch.tensor([act])
    tensors.update(extra or {})
    _save(tmp_path / "model.safetensors", tensors)
    return tensors


@pytest.fixture(autouse=True)
def _forbid_grouped_and_draft_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("NVFP4 loading MUST use checkpoint packing without grouped experts or a draft head")

    monkeypatch.setattr(weights.grouped, "make", forbidden)
    monkeypatch.setattr(weights, "quantize4", forbidden)
    monkeypatch.setattr(table, "_refuse_production", forbidden)
    # linear._ext builds experts.cu; checkpoint packing uses checkpoint._ext instead.
    monkeypatch.setattr(fp4, "_ext", forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("presplit", [False, True])
def test_e4_load_prefetches_checkpoint_keys_and_keeps_bf16_arms(tmp_path, monkeypatch, rank, presplit):
    source = _checkpoint(tmp_path)
    model_dir = tmp_path
    if presplit:
        model_dir = tmp_path / "rank"
        model_dir.mkdir()
        (model_dir / "config.json").write_text((tmp_path / "config.json").read_text())
        split_file(tmp_path / "model.safetensors", model_dir, rank)
    queued = []
    real = RankReader.prefetch

    def record(self, names, device=None):
        queued.append((list(names), device))
        return real(self, names, device)

    monkeypatch.setattr(RankReader, "prefetch", record)
    loaded = weights.load(model_dir, rank=rank, mtp=False)
    assert loaded.cfg.quant == "modelopt" and loaded.meta["layers"] == [0, 1, 2]
    assert loaded.mtp is None and loaded.draft_head is None
    assert loaded.rank == rank and loaded.world == 2
    assert torch.equal(loaded.embed.cpu(), source[PREFIX + "embed_tokens.weight"])
    assert isinstance(loaded.head, B16)
    assert torch.equal(loaded.head.weight.cpu(), source["lm_head.weight"][rank * 64:(rank + 1) * 64])
    assert loaded.norm.dtype == loaded.layers[0].in_norm.dtype == torch.bfloat16
    assert isinstance(loaded.layers[0].kda.proj, B16)
    assert isinstance(loaded.layers[0].mlp.gu, B16)
    assert isinstance(loaded.layers[1].dsa.proj, B16)
    assert isinstance(loaded.layers[1].dsa.q_b, B16)
    assert isinstance(loaded.layers[1].dsa.kv_k, B16)
    assert isinstance(loaded.layers[1].dsa.kv_v, B16)
    assert isinstance(loaded.layers[1].dsa.index.kw, B16)
    assert loaded.layers[1].moe.router.dtype == torch.bfloat16
    assert loaded.layers[1].moe.bias.dtype == torch.float32
    assert {name for names, _ in queued for name in names} == set(table.prefetch_names(1, 4) +
                                                                 table.prefetch_names(2, 4))
    assert all(device == loaded.device for names, device in queued if names)

    reader = RankReader(model_dir, rank)
    try:
        for lw in loaded.layers[1:]:
            routed = lw.moe.experts
            assert isinstance(routed, table.RoutedTable) and routed.experts == 4
            assert routed.combine_weight == 1.0 and len(routed.linears) == 12
            assert lw.moe.shared is None  # Shared tensors belong to the table once.
            for slot, linear in enumerate(routed.linears):
                e, col = divmod(slot, 3)
                proj = table.PROJECTIONS[col]
                assert int(routed.words_ptr[e, col]) == linear.words.data_ptr()
                assert int(routed.bs_ptr[e, col]) == linear.bs.data_ptr()
                name = PREFIX + f"layers.{lw.index}.mlp.experts.{e}.{proj}."
                factor, act = FACTORS[proj]
                expected = fp4.Fp4Linear.from_checkpoint(reader.get(name + "weight").cuda(),
                                                       reader.get(name + "weight_scale").cuda(), factor, act=act)
                assert torch.equal(linear.words, expected.words)
                assert torch.equal(linear.bs, expected.bs)
                assert (linear.scale, linear.act) == (factor, act)
                assert (int(routed.n[e, col]), int(routed.k[e, col])) == (expected.n, expected.k)
                assert float(routed.alpha[e, col]) == table.rounded_alpha(act, factor)
            for proj in table.PROJECTIONS:
                name = PREFIX + f"layers.{lw.index}.mlp.shared_experts.{proj}.weight"
                assert torch.equal(routed.shared[proj].cpu(), reader.get(name))
    finally:
        reader.close()


def test_e4_numeric_fixture_uses_the_loaded_addresses(tmp_path):
    _checkpoint(tmp_path)
    loaded = weights.load(tmp_path, rank=0, mtp=False)
    routed = loaded.layers[1].moe.experts
    generator = torch.Generator().manual_seed(61)
    for slot, linear in enumerate(routed.linears):
        e, col = divmod(slot, 3)
        x = torch.randn((1, linear.k), generator=generator).to(torch.bfloat16).cuda()
        expected = checkpoint.matmul(checkpoint.A4, x, linear, f32=True)
        out = torch.full_like(expected, float("nan"))
        assert nvfp4_dispatch.projection(checkpoint.quant4(x, linear.act), routed, e, col, out) is out
        assert torch.isfinite(out).all() and out.abs().max() > 0
        assert torch.equal(out.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("name,match", [
    ("model.draft_head.weight", "draft head"),
])
def test_forbidden_tensor_raises_before_expert_allocation(tmp_path, monkeypatch, name, match):
    _checkpoint(tmp_path, extra={name: torch.ones(1, dtype=torch.bfloat16)})
    closed = []
    real_close = RankReader.close

    def forbidden(*args, **kwargs):
        pytest.fail("forbidden names MUST raise before reads, prefetch, or expert allocation")

    def close(self):
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(RankReader, "get", forbidden)
    monkeypatch.setattr(RankReader, "prefetch", forbidden)
    monkeypatch.setattr(fp4.Fp4Linear, "from_checkpoint", forbidden)
    monkeypatch.setattr(RankReader, "close", close)
    with pytest.raises(ValueError, match=match):
        weights.load(tmp_path, rank=0, mtp=False)
    assert len(closed) == 1


def test_layer_45_in_the_index_is_not_loaded(tmp_path, monkeypatch):
    name = PREFIX + "layers.45.mlp.experts.0.gate_proj.weight"
    _checkpoint(tmp_path, extra={name: torch.ones(1, dtype=torch.bfloat16)})
    seen = []
    real = RankReader.get

    def record(self, key, *args, **kwargs):
        seen.append(key)
        return real(self, key, *args, **kwargs)

    monkeypatch.setattr(RankReader, "get", record)
    loaded = weights.load(tmp_path, rank=0, mtp=False)
    assert all(".layers.45." not in key for key in seen)
    assert loaded.mtp is None


def test_dense_layer_with_scales_is_packed(tmp_path):
    _checkpoint(tmp_path, dense_nvfp4=True)
    loaded = weights.load(tmp_path, rank=0, mtp=False)
    packed = loaded.layers[0].dense_nvfp4
    assert loaded.layers[0].mlp is None
    assert packed.layer == 0 and packed.experts == 1 and len(packed.linears) == 3


@pytest.mark.parametrize("failure", ["packing", "head"])
def test_failed_load_frees_its_owners_and_preserves_caller_storage(tmp_path, monkeypatch, failure):
    _checkpoint(tmp_path)
    # Warm compilation separately so extension initialization is outside the memory baseline.
    checkpoint._ext()
    caller = torch.full((128, 128), 0.25, dtype=torch.bfloat16, device="cuda")
    caller_ptr = caller.data_ptr()
    retained = weights.load(tmp_path, rank=1, mtp=False)
    retained_linear = retained.layers[1].moe.experts.linears[0]
    retained_ptr = retained_linear.words.data_ptr()
    owners = []
    real_pack, real_get = fp4.Fp4Linear.from_checkpoint, RankReader.get
    calls = 0

    def pack(*args, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "packing" and calls == 14:  # One finished table and one partially packed table.
            raise RuntimeError("injected pack failure")
        result = real_pack(*args, **kwargs)
        owners.extend((weakref.ref(result.words), weakref.ref(result.bs)))
        return result

    def get(self, name):
        if failure == "head" and name == "lm_head.weight":
            raise RuntimeError("injected head failure")
        return real_get(self, name)

    monkeypatch.setattr(fp4.Fp4Linear, "from_checkpoint", pack)
    monkeypatch.setattr(RankReader, "get", get)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with pytest.raises(RuntimeError, match="injected") as error:
        weights.load(tmp_path, rank=0, mtp=False)
    # Keep the exception and its traceback alive: they MUST NOT retain the failed load's buffers.
    assert error.value is not None
    gc.collect()
    torch.cuda.synchronize()
    assert owners and all(ref() is None for ref in owners)
    assert torch.cuda.memory_allocated() == before
    assert caller.data_ptr() == caller_ptr and bool((caller == 0.25).all())
    assert retained_linear.words.data_ptr() == retained_ptr
    assert int(retained.layers[1].moe.experts.words_ptr[0, 0]) == retained_ptr


def test_e288_table_addresses_need_only_tiny_buffers():
    words = table.empty_address_table(288, device="cuda")
    scales = torch.empty_like(words)
    tiny_words = [torch.empty(1, dtype=torch.int32, device="cuda") for _ in range(288 * 3)]
    tiny_scales = [torch.empty(1, dtype=torch.uint8, device="cuda") for _ in range(288 * 3)]
    table.bind_addresses(words, tiny_words)
    table.bind_addresses(scales, tiny_scales)
    assert words.shape == scales.shape == (288, 3)
    assert words.view(-1).tolist() == [t.data_ptr() for t in tiny_words]
    assert scales.view(-1).tolist() == [t.data_ptr() for t in tiny_scales]


def test_weights_nbytes_counts_fp4_and_shared_backing_buffers_once():
    staging = fp4.Staging()
    staging.w8 = torch.empty(64, dtype=torch.uint8, device="cuda")
    staging.s8 = torch.empty(32, dtype=torch.bfloat16, device="cuda")
    words = torch.empty((2, 1, 8, 32, 2), dtype=torch.int32, device="cuda")
    scales = torch.empty((2, 1, 64, 4), dtype=torch.uint8, device="cuda")
    a = fp4.Fp4Linear(words, scales, 0.125, 128, 64, staging=staging, act=0.5)
    b = fp4.Fp4Linear(words[1:], scales[1:], 0.25, 64, 64, staging=staging, act=0.5)
    held = weights.Weights.__new__(weights.Weights)
    held.embed = ()
    held.layers = [a, a, b]
    held.norm = held.head = torch.empty(0, device="cuda")
    held.draft_head = held.mtp = None
    assert a.words.data_ptr() == words.data_ptr()
    assert b.words.untyped_storage().data_ptr() == a.words.data_ptr()
    assert b.words.data_ptr() != a.words.data_ptr()
    assert held.nbytes() == words.nbytes + scales.nbytes + staging.nbytes()
