# GLM-5.3-Flash NVIDIA NVFP4 Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task.
>
> **This commit is the plan only.** Do not implement NVFP4, do not pull weights, and do not start a serve in this commit.

**Goal:** Teach TensorFold's `glm5_next` CUDA path to load and serve `nvidia/GLM-5.3-Flash-NVFP4`, then prove a short greedy completion matches vLLM on those same weights before any speed claim.

**Architecture:** Reuse `tensorfold.cuda.nvfp4` (`format.scheme`, `Fp4Linear.from_checkpoint`, the expert kernel) the way `qwen3_5/cuda/nvfp4_load.py` already does. Do not write a new GEMM. Add a GLM-specific loader and wire the existing GLM forward to it. The first cut is the official ModelOpt export only. GLM CUDA is two-rank only, and every existing NVFP4 family refuses `--tp 2`. Qualifying two-rank NVFP4 is part of this work, not a follow-up.

**Tech stack:** TensorFold 0.6.2 (Apache-2.0), base `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21` (`release: TensorFold 0.6.2`, 2026-10-02). PyTorch CUDA. Existing NVFP4 kernels: FP4×FP4 on SM 12.x, W4A16 fallback where the GPU has no block-scaled FP4 MMA. Upstream parent: [ashhart/TensorFold](https://github.com/ashhart/TensorFold).

**Pinned tree:** Re-read upstream before the first code task. This repo moved the morning the plan was written. If `main` is no longer `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`, rebase this branch onto the new pin and re-check every path cited below before editing.

---

## Decision record

This plan exists because the published GLM CUDA recipe does not read the quant we treat as the quality bar, and the published speed claims are same-checkpoint wins against vLLM on other families.

Quality bar, in order:

1. `nvidia/GLM-5.3-Flash-NVFP4` — official ModelOpt W4A4. **First implementation target.** The existing NVFP4 reader already speaks ModelOpt.
2. `RedHatAI/GLM-5.3-Flash-NVFP4` — compressed-tensors. Same quality class. **Not this plan's implementation.** A prior same-engine comparison on vLLM found this tree ahead of the official NVIDIA export on quality judgments already run. That result does not answer a different engine on the NVIDIA weights, and it is not a reason to skip this port. It is a reason to keep RedHat as the second checkpoint, after the ModelOpt loader works, and to judge TensorFold+NVIDIA against vLLM+NVIDIA before comparing either to the RedHat vLLM serve.
3. EXL3 / TR3 4bpw is **out of scope**. It is a different codebook, not a stand-in for NVFP4. Upstream still marks GLM EXL3 speed, capacity, and long-context qualification as TBD.

NVIDIA-first is the natural order for this codebase, not a claim that the official export is the better quant. `src/tensorfold/cuda/nvfp4/format.py` already accepts `quant_method` of `modelopt` or `compressed-tensors`. The GLM family never calls that reader.

Do not displace the existing multi-request vLLM serve to write this plan, to run CPU tests, or to pull weights. An exclusive window is allowed only after a logit-match harness exists and that window is explicitly approved. A single-stream win does not replace a server that already runs more than one request at a time.

---

## What upstream already does

Fork parent and pin:

- https://github.com/ashhart/TensorFold
- https://github.com/ashhart/TensorFold/commit/56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21

GLM recipe, quoted from `docs/recipes/glm-5.3-flash.md` on that pin:

> No NVFP4 checkpoint of it is read.

CUDA checkpoint table, `docs/recipes/cuda.md`:

| Family | NVFP4 | EXL3 | MLX 4-bit |
| --- | --- | --- | --- |
| GLM-5.3-Flash | not read | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (re-host `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`), two ranks, experimental | two ranks |

The CUDA loader rejects anything else. `src/tensorfold/families/glm5_next/cuda/weights.py` (`load`, around the `cfg.quant` check):

```python
if cfg.quant not in ("mlx", "exl3"):
    raise ValueError(f"GLM-5.3-Flash's CUDA engine reads MLX 4-bit or EXL3 checkpoints, not {cfg.quant}")
```

`Config.read` sets `quant` from `quant_method` (default `"mlx"`). A ModelOpt config therefore dies at that raise, before any tensor is read.

GLM CUDA is two-rank only. `src/tensorfold/families/glm5_next/cuda/engine.py` documents the engine as two ranks. `src/tensorfold/families/glm5_next/__init__.py` requires `--tp 2` and `--master`. It serves one request at a time (`docs/recipes/glm-5.3-flash.md`). A checkpoint with neither an MTP head nor a supplied DFlash2 model is refused unless drafts are disabled.

Existing NVFP4 is real, and it is not wired to GLM:

- Reader and scheme detection: `src/tensorfold/cuda/nvfp4/format.py` (`METHODS = ("modelopt", "compressed-tensors")`, `scheme()`, `config_block()`, `is_quantized()`).
- Dense linear: `src/tensorfold/cuda/nvfp4/linear.py` (`Fp4Linear.from_checkpoint`).
- Experts: `src/tensorfold/cuda/nvfp4/experts.py`. Shape is generic `[E, N, K]`. Tensor names and the TP split are not. Do not assume Qwen packing.
- Working ModelOpt loader to copy from, not to import as a GLM loader: `src/tensorfold/families/qwen3_5/cuda/nvfp4_load.py` (`load_nvfp4`). It already branches ModelOpt (`weight_scale_2`) vs compressed-tensors (`weight_global_scale`, reciprocal).
- Family recipe for a new CUDA path: `docs/recipes/adding-a-cuda-family.md`. Exactness is against the same engine's serial reference, separate from quality against a trusted model. Set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0` for an fp32 quality reference.

Two-rank NVFP4 is unqualified on every family that already speaks NVFP4. That is the actual engineering risk, not "NVFP4 as a format."

`src/tensorfold/families/qwen3_5/cuda/engine.py`:

```python
if (exl3 or nvfp4) and tp != 1:
    raise ValueError(
        f"{'EXL3 packs' if exl3 else 'NVFP4 checkpoints'} of Qwen3.8-27B run on one GPU: drop "
        "--tp 2, or serve the MLX checkpoint (Vontra/Qwen3.8-27B-MLX-4bit) on two"
    )
```

`tests/cuda/test_flashnext_nvfp4_loader.py::test_an_nvfp4_checkpoint_refuses_two_ranks` expects Flash Next NVFP4 on `--tp 2` to raise `one GPU` before NCCL starts. `CHANGELOG.md` (0.3.6.3) says two ranks, `--ple-on-ssd`, and images on NVFP4 checkpoints stop at startup until qualified. GLM cannot drop `--tp 2`. Do not copy the Qwen refusal. The work is a group-16-safe TP split plus a two-rank exactness check.

---

## Why a speed bet is reasonable, and what it is not

These are same-checkpoint comparisons published by TensorFold. They are not GLM-5.3-Flash NVFP4 measurements. GLM NVFP4 has no TensorFold number because the loader refuses it.

Verified on the pin:

- `CHANGELOG.md` 0.6.1: `nvidia/Qwen3.8-27B-NVFP4` on an RTX PRO 6000 at its 250 W limit, one stream, decodes at 1.4–2.0× vLLM. Prompts fill at 0.95–0.97× vLLM.
- `CHANGELOG.md` 0.3.6.3: on one Spark, Flash Next NVFP4 decodes at 1.13–1.52× vLLM on the same checkpoint.
- `CHANGELOG.md` 0.5.0, in the Spark long-context note: on the same NVFP4 weights, prefill runs 1.16–1.27× vLLM from 32k to 255k. Quote it as written. It is not a GLM result.
- `docs/recipes/qwen3.8-27b.md`, `nvidia/Qwen3.8-27B-NVFP4` vs vLLM MTP=3 on that NVFP4 checkpoint (decode, tok/s):

| Cell | TensorFold NVFP4 | vLLM MTP=3 |
| --- | ---: | ---: |
| Code, sampled | 47.1 | 23.4 |
| Chat, sampled | 38.2 | 25.4 |
| Code, greedy | 47.2 | 25.8 |
| Chat, greedy | 38.3 | 24.7 |

The bet is that this gap can transfer to GLM once the loader and the two-rank split exist. It is not a promised 1.5×. Speed only counts after a logit match on `nvidia/GLM-5.3-Flash-NVFP4`.

Published GLM CUDA numbers are the wrong quant. `docs/recipes/glm-5.3-flash.md`, two DGX Spark GB10s, `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, MTP only, one request at a time, `--context 262144`:

| Prompt | Prompt reading | First token | Decode |
| --- | ---: | ---: | ---: |
| 32,770 | 1,138 tok/s | 29 s | 50.9 tok/s |
| 131,074 | 898 tok/s | 146 s | 47.2 tok/s |
| 261,906 | 849 tok/s | 309 s | 31.9 tok/s |

Do not use that table as the success bar. Do not spark-eval an MLX or EXL3 TensorFold serve against the NVFP4 vLLM serve and call it this project.

---

## Checkpoint map

Fetched 2026-10-02 from the Hugging Face config, not from a local weight pull.

Official export: https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4

`config.json` facts that the loader must honor:

- `model_type`: `glm5_next`
- `architectures`: `Glm5NextForConditionalGeneration`
- `quantization_config.quant_method`: `modelopt`
- `quantization_config.quant_algo`: `NVFP4`
- `producer.name`: `modelopt`
- `producer.version`: `0.47.0.dev393+ga4bc45b30.d20260828`
- Weights and input activations: 4-bit float, group size 16, static (not dynamic)
- `kv_cache_scheme`: static FP8 (`num_bits` 8, `type` float). This may matter for a vLLM logit match. Do not assume the existing bf16 latent cache is equivalent. Inspect what vLLM actually does with this field before writing a cache. If vLLM uses FP8 KV, a bf16-cache serve will fail the logit gate. Do not loosen the gate to hide that.

Text config (language model):

- `num_hidden_layers`: 45
- `num_nextn_predict_layers`: 1
- `first_k_dense_replace`: 3 (layers 0–2 are dense MLP)
- `hidden_size`: 4096
- `intermediate_size`: 12288 (dense MLP)
- `moe_intermediate_size`: 2048
- `n_routed_experts`: 288
- `n_shared_experts`: 1
- `num_experts_per_tok`: 8
- `num_attention_heads` / `num_key_value_heads`: 64
- `vocab_size`: 154880
- `max_position_embeddings`: 1048576

`quantization_config.ignore` is the quantization map. Ignored modules are not NVFP4. On this export they are:

- `lm_head`
- `model.language_model.embed_tokens`
- `self_attn*` on layers 0–44
- `mlp.gate` and `shared_experts*` on the MoE layers (3–44 in the ignore list)
- `model.visual*`
- `model.language_model.layers.45*` and `model.layers.45*` (the MTP layer)

So the NVFP4 surface, until a safetensors index says otherwise, is:

- Dense MLP linears on layers 0–2 (`intermediate_size` 12288). Those layers ignore attention only.
- Routed experts on MoE layers (`n_routed_experts` 288, `moe_intermediate_size` 2048). They are not in the ignore list.

Left on the existing BF16 path:

- Attention, indexer, KDA, router (`mlp.gate`), shared experts, embeddings, `lm_head`, vision tower.
- Layer 45 / MTP, if the tensors exist. Ignore means "not NVFP4", not "absent". Confirm against `model.safetensors.index.json` before assuming drafts. First serve uses `--no-drafts` or `--drafter none` until that census is done.

Do not pull `incoai/GLM-5.3-Flash-DFlash2`. Upstream documents CC BY-NC-ND 4.0 terms. This project does not accept that license.

Second checkpoint, not loaded in this plan: https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4

- `quant_method`: `compressed-tensors`
- `format`: `mixed-precision`
- The Qwen loader already treats compressed-tensors global scales as reciprocals. Reuse that branch later. Do not start the RedHat loader in this branch.

Rejected as the target quant:

- https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw
- https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw
- https://huggingface.co/Vontra/GLM-5.3-Flash-MLX-4bit-MTP (portable, already served, not the quality bar)

Reference loader checkpoint, already read by TensorFold: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4

---

## Non-goals

- No new NVFP4 GEMM, packer, or scale format. Call `Fp4Linear.from_checkpoint` and the existing expert kernel.
- No EXL3 / TR3 work. Do not "finish" the experimental GLM EXL3 path as a stepping stone.
- No RedHat compressed-tensors loader in this branch.
- No Apple / MLX NVFP4. The Mac path has no NVFP4 lane for this model. A 13-inch Mac cannot hold it. Do not block the CUDA port on a Mac serve.
- No vision serve. The tower is in the ignore list. Text serve skips it the way the Qwen NVFP4 loader skips `model.visual`.
- No DFlash2 pull, and no MTP policy work, until the index proves an MTP head exists and the BF16 path can load it.
- No concurrency work. GLM CUDA serves one request at a time. Do not advertise `--parallel` for this port.
- No weight pull and no exclusive serve window until the tasks below say so, and not without a separate approval.
- No comparison of TensorFold MLX or EXL3 against the NVFP4 vLLM serve.

---

## Stop conditions

Stop and report. Do not tune for speed past any of these.

- The safetensors index names a non-ignored `Linear` the loader cannot map.
- A two-rank split cuts an NVFP4 group of 16, or the kernel tile, in half.
- A short greedy completion on `nvidia/GLM-5.3-Flash-NVFP4` diverges from vLLM on the same weights, same prompt, same sampling (greedy). Divergence is a failed port, not a benchmark.
- The only way to continue is to bounce or delete the existing vLLM serve before the logit harness exists.
- The work starts requiring a new GEMM instead of the kernels in `src/tensorfold/cuda/nvfp4/`.

---

## Files

Create:

- `src/tensorfold/families/glm5_next/cuda/nvfp4_load.py`
- `tests/cuda/test_glm5_nvfp4_loader.py`
- `tests/cuda/test_glm5_nvfp4_config.py` (CPU, no weights, no GPU)

Modify, only after the census task:

- `src/tensorfold/families/glm5_next/cuda/weights.py` — admit `modelopt` / NVFP4, delegate to `nvfp4_load.py`. Keep the MLX and EXL3 branches. Do not fold NVFP4 tensor names into `trip()` / `make_q4`.
- `src/tensorfold/families/glm5_next/cuda/forward.py` and the MoE call site in `weights.py` — call `Fp4Linear` for dense MLP layers 0–2 and the existing NVFP4 expert kernel for routed experts. Attention, gate, shared experts, embeddings, head stay on the current BF16 path.
- `src/tensorfold/families/glm5_next/cuda/split.py` — TP2 split must land on a multiple of 16 (NVFP4 group), not only the MLX group of 64.
- `docs/recipes/glm-5.3-flash.md` and the GLM row of `docs/recipes/cuda.md` — only after a real load, and only to say what is actually qualified. Do not claim two-rank or long-context qualification before the tests exist.
- `tests/cuda/test_qwen27_nvfp4.py` and `tests/cuda/test_flashnext_nvfp4_loader.py` — do not weaken their `--tp 2` refusals. GLM is the family that must grow a two-rank path. Qwen and Flash Next stay one-GPU until upstream qualifies them.

Read before editing, on the pin this plan was written against:

- `docs/recipes/glm-5.3-flash.md`
- `docs/recipes/cuda.md`
- `docs/recipes/adding-a-cuda-family.md`
- `docs/recipes/qwen3.8-27b.md`
- `src/tensorfold/families/glm5_next/cuda/weights.py`
- `src/tensorfold/families/glm5_next/cuda/forward.py`
- `src/tensorfold/families/glm5_next/cuda/split.py`
- `src/tensorfold/families/qwen3_5/cuda/nvfp4_load.py`
- `src/tensorfold/cuda/nvfp4/format.py`
- `src/tensorfold/cuda/nvfp4/linear.py`
- `src/tensorfold/cuda/nvfp4/experts.py`
- `tests/cuda/test_nvfp4_checkpoint.py`
- `tests/cuda/test_flashnext_nvfp4_loader.py`

---

### Task 1: Pin the tree and record the weight census

**Objective:** Know every tensor the NVIDIA export actually stores, and which of those `format.scheme` will call NVFP4, before any loader code.

**Files:**

- Create: `docs/plans/notes/nvidia-glm-nvfp4-index.md` (this note is allowed; it is a census, not a serve)
- Do not modify loader code in this task.

**Step 1: Confirm the pin**

```bash
git rev-parse HEAD
git merge-base --is-ancestor 56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21 HEAD && echo PIN_OK
```

Expected: `PIN_OK`. If the branch was rebased, re-read the files listed above and update this plan's line citations before Task 2.

**Step 2: Census the public index, without downloading shards**

```bash
curl -fsSL --max-time 60 \
  -o /tmp/nvidia-glm-nvfp4-index.json \
  https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/resolve/main/model.safetensors.index.json
python3 - << 'PY'
import json
from collections import Counter
idx = json.load(open("/tmp/nvidia-glm-nvfp4-index.json"))
wm = idx["weight_map"]
print("tensors", len(wm))
print("has_mtp", any(".45." in n or n.startswith("model.layers.45") or "nextn" in n or "mtp" in n for n in wm))
needles = ("self_attn", "mlp.experts", "shared_experts", "mlp.gate", "embed_tokens", "lm_head", "visual")
for needle in needles:
    print(needle, sum(needle in n for n in wm))
suffixes = Counter(n.rsplit(".", 1)[-1] for n in wm)
print("suffixes", suffixes.most_common(20))
PY
```

Expected: a non-zero `mlp.experts` count, a non-zero dense-MLP count on layers 0–2, and an explicit `has_mtp` true or false. Paste the counts into `docs/plans/notes/nvidia-glm-nvfp4-index.md`. If `mlp.experts` is zero, stop. The ignore-list reading in this plan is wrong.

**Step 3: Record shard bytes**

```bash
python3 - << 'PY'
import json
from collections import Counter
idx = json.load(open("/tmp/nvidia-glm-nvfp4-index.json"))
print("shards", len(set(idx["weight_map"].values())))
meta = idx.get("metadata") or {}
print("metadata_keys", sorted(meta))
PY
```

Write the total size if the index metadata has it. If it does not, sum `Content-Length` from the Hugging Face file listing and record that. Do not pull the shards in this task. Do not delete any existing serve's weights to make room.

**Step 4: Commit**

```bash
git add docs/plans/notes/nvidia-glm-nvfp4-index.md
git commit -m "docs: census nvidia GLM-5.3-Flash NVFP4 tensors"
```

---

### Task 2: CPU test that the NVIDIA config is NVFP4 and GLM still refuses it

**Objective:** Lock the current refusal, and lock the scheme helper's reading of this config, before changing the loader.

**Files:**

- Create: `tests/cuda/test_glm5_nvfp4_config.py`
- Test: that file

**Step 1: Write the failing / locking test**

```python
"""CPU checks for the NVIDIA GLM-5.3-Flash NVFP4 config. No weights, no GPU."""

import json
from pathlib import Path

import pytest

from tensorfold.cuda.nvfp4.format import config_block, scheme

NVIDIA_QUANT = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "config_groups": {
        "group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"],
        }
    },
    "ignore": ["lm_head", "model.language_model.embed_tokens", "model.visual*"],
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    "producer": {"name": "modelopt", "version": "0.47.0.dev393+ga4bc45b30.d20260828"},
}


def test_modelopt_block_is_recognized():
    block = config_block({"model_type": "glm5_next", "quantization_config": NVIDIA_QUANT})
    assert block is not None
    assert block["quant_method"] == "modelopt"
    assert block["quant_algo"] == "NVFP4"
    assert block["config_groups"]["group_0"]["weights"]["group_size"] == 16


def test_scheme_names_packed_nvfp4():
    assert scheme({"weight_packed": ("U8", [128, 64]), "weight_scale": ("F8_E4M3", [128, 4])}) == "nvfp4"


def test_glm_cuda_loader_still_refuses_modelopt(tmp_path: Path):
    """Delete this test in the same commit that admits modelopt. Until then it must fail closed."""
    cfg = {
        "model_type": "glm5_next",
        "text_config": {"hidden_size": 4096, "num_hidden_layers": 45},
        "quantization_config": NVIDIA_QUANT,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from tensorfold.families.glm5_next.cuda.weights import load

    with pytest.raises(ValueError, match="not modelopt|not nvfp4|MLX 4-bit or EXL3"):
        load(tmp_path, rank=0)
```

The third test's match is the current raise (`not {cfg.quant}`). If `Config.read` names the method `modelopt`, the message contains `not modelopt`. If it names something else, fix the test's match to the real message and record that name in the census note. Do not change `weights.py` in this task.

**Step 2: Run it**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_config.py -q
```

Expected: the scheme tests pass. `test_glm_cuda_loader_still_refuses_modelopt` passes if the loader raises, or fails if `Config.read` rejects the fixture before `load`'s quant check. Either failure mode is information. Make the fixture complete enough that execution reaches the quant raise, using the fields `Config.read` actually requires. Read `Config.read` in `weights.py` and add those fields. Do not download a checkpoint to satisfy the fixture.

**Step 3: Commit**

```bash
git add tests/cuda/test_glm5_nvfp4_config.py
git commit -m "test: lock GLM CUDA refusal of ModelOpt NVFP4"
```

---

### Task 3: Admit the config without loading tensors

**Objective:** `load()` stops saying ModelOpt is an unknown quant, and instead says the NVIDIA GLM tensors are not wired yet. MLX and EXL3 still load as they do today.

**Files:**

- Modify: `src/tensorfold/families/glm5_next/cuda/weights.py` (the `cfg.quant not in ("mlx", "exl3")` raise only)
- Modify: `tests/cuda/test_glm5_nvfp4_config.py`
- Test: `tests/cuda/test_glm5_nvfp4_config.py`

**Step 1: Change the locking test into the new refusal**

Replace `test_glm_cuda_loader_still_refuses_modelopt` so it expects a not-wired error, not the old quant error:

```python
def test_glm_cuda_loader_names_nvfp4_but_does_not_load_it_yet(tmp_path: Path):
    cfg = {
        "model_type": "glm5_next",
        "text_config": {"hidden_size": 4096, "num_hidden_layers": 45},
        "quantization_config": NVIDIA_QUANT,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from tensorfold.families.glm5_next.cuda.weights import load

    with pytest.raises(ValueError, match="NVFP4 tensors are not wired"):
        load(tmp_path, rank=0)
```

Keep the fixture fields that Task 2 found `Config.read` requires.

**Step 2: Run to see it fail**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_config.py::test_glm_cuda_loader_names_nvfp4_but_does_not_load_it_yet -q
```

Expected: FAIL. The message is still the old `MLX 4-bit or EXL3` raise, or `Config.read` never returns `quant="modelopt"`.

**Step 3: Minimal admit**

In `Config.read`, keep storing `quant_method` lowercased. In `load()`, replace the hard `("mlx", "exl3")` rejection with:

```python
if cfg.quant == "modelopt":
    raise ValueError("GLM-5.3-Flash NVFP4 tensors are not wired")
if cfg.quant not in ("mlx", "exl3"):
    raise ValueError(
        f"GLM-5.3-Flash's CUDA engine reads MLX 4-bit, EXL3, or NVIDIA NVFP4, not {cfg.quant}"
    )
```

Do not call `nvfp4_load` yet. Do not accept `compressed-tensors` here. RedHat must still hit the second raise.

**Step 4: Run**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_config.py tests/test_glm5_next_family.py -q
```

Expected: PASS. If `test_glm5_next_family.py` asserts the old error string, update that assertion in the same commit and say so in the message. Do not weaken an EXL3 or MLX test to make NVFP4 pass.

**Step 5: Commit**

```bash
git add src/tensorfold/families/glm5_next/cuda/weights.py tests/cuda/test_glm5_nvfp4_config.py
git commit -m "feat: name NVIDIA NVFP4 in the GLM CUDA loader without reading it"
```

---

### Task 4: Map one dense MLP linear onto `Fp4Linear`

**Objective:** Layers 0–2 dense MLP projections become `Fp4Linear` objects built by `Fp4Linear.from_checkpoint`, using the Qwen scale convention. No forward call yet.

**Files:**

- Create: `src/tensorfold/families/glm5_next/cuda/nvfp4_load.py`
- Create: `tests/cuda/test_glm5_nvfp4_loader.py`
- Modify: `src/tensorfold/families/glm5_next/cuda/weights.py` (replace the Task 3 `not wired` raise with a call, still only after a synthetic or real index says the projection exists)

**Step 1: Write a synthetic-tensor test**

Follow `tests/cuda/test_nvfp4_checkpoint.py` and `tests/cuda/test_flashnext_nvfp4_loader.py` for how they build a tiny ModelOpt linear (`weight` or `weight_packed` as `U8`, `weight_scale` as `F8_E4M3`, `weight_scale_2` as the ModelOpt global scale, `input_scale` present). Do not invent a second packing. Copy the fixture helper those tests already use.

The new test:

- Builds one GLM-shaped name from the Task 1 census (expected prefix `model.language_model.layers.0.mlp.`, suffixes `gate_proj` / `up_proj` / `down_proj` or whatever the index actually uses — use the index, not this sentence, if they differ).
- Calls the new loader function on that directory.
- Asserts the returned object is an `Fp4Linear`.
- Asserts `format.scheme` on that projection is `nvfp4`.
- Asserts a missing `weight_scale_2` raises `global scale`, matching `nvfp4_load.py`'s Qwen error.

**Step 2: Run to see it fail**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_loader.py -q
```

Expected: FAIL — import error or function missing.

**Step 3: Implement the mapper only**

In `nvfp4_load.py`:

- Call `format.config_block` and refuse anything whose `quant_method` is not `modelopt`.
- Call `format.scheme` per projection. `nvfp4` goes to `Fp4Linear.from_checkpoint(weight, weight_scale, global_scale, act=input_scale)`.
- ModelOpt global scale is `weight_scale_2`, passed through as stored. Do not take the reciprocal. That reciprocal is the compressed-tensors branch in `qwen3_5/cuda/nvfp4_load.py`. Leave it unimplemented and raise `compressed-tensors is a later checkpoint` if seen.
- `bf16` projections return the existing BF16 linear type the GLM forward already calls. Do not quantize them.
- Skip `model.visual` the way `qwen3_5/cuda/nvfp4_load.py` `skipped()` skips vision.
- Do not upload experts in this task. If the census name is an expert tensor, raise `routed experts are not wired`.

**Step 4: Run**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_loader.py tests/cuda/test_glm5_nvfp4_config.py -q
```

Expected: PASS. GPU skip is acceptable only for the assertion that constructs a CUDA tensor, and the test must say so with the same `pytest.mark.skipif(not torch.cuda.is_available())` pattern as `test_flashnext_nvfp4_loader.py`. The config test stays CPU.

**Step 5: Commit**

```bash
git add src/tensorfold/families/glm5_next/cuda/nvfp4_load.py tests/cuda/test_glm5_nvfp4_loader.py
git commit -m "feat: map one GLM dense MLP projection to Fp4Linear"
```

---

### Task 5: TP2 split respects group 16

**Objective:** A two-rank split of an NVFP4 row never cuts a group of 16, and the existing MLX group-64 split still passes.

**Files:**

- Modify: `src/tensorfold/families/glm5_next/cuda/split.py`
- Test: `tests/cuda/test_glm5_nvfp4_loader.py` or a new `tests/cuda/test_glm5_nvfp4_split.py` if the split test needs no CUDA

**Step 1: Write the failing test**

```python
def test_nvfp4_tp2_split_keeps_groups_of_16():
    from tensorfold.families.glm5_next.cuda.split import nvfp4_row_split

    # 288 experts, intermediate 2048, hidden 4096 are the checkpoint's sizes.
    # A legal split returns equal row ranges whose lengths are multiples of 16.
    left, right = nvfp4_row_split(n_rows=2048, world=2, group=16)
    assert left.stop - left.start == right.stop - right.start
    assert (left.stop - left.start) % 16 == 0
    assert (right.stop - right.start) % 16 == 0
    assert left.start == 0 and right.stop == 2048 and left.stop == right.start
```

Add the function only if `split.py` does not already expose an equivalent. If it does, call that function in the test instead of adding `nvfp4_row_split`. Read `split.py` first. Do not duplicate a splitter that already exists for group 64; parameterize the group size.

**Step 2: Run to see it fail**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_split.py::test_nvfp4_tp2_split_keeps_groups_of_16 -q
```

Expected: FAIL — function missing, or a group-64-only splitter rejects group 16.

**Step 3: Implement the smallest splitter change**

Reject a row count that is not a multiple of `group * world`. Do not pad silently. A pad would change the matmul shape and hide a bad checkpoint.

**Step 4: Run the existing GLM split tests plus the new one**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_split.py tests/test_glm5_layouts.py -q
```

Expected: PASS. If `tests/test_glm5_layouts.py` does not cover the split, also run whatever test file imports `split.py` (search `from tensorfold.families.glm5_next.cuda.split` and run those).

**Step 5: Commit**

```bash
git add src/tensorfold/families/glm5_next/cuda/split.py tests/cuda/test_glm5_nvfp4_split.py
git commit -m "feat: split GLM NVFP4 rows on groups of 16"
```

---

### Task 6: Routed experts use the existing expert kernel

**Objective:** One MoE layer's routed experts load as the existing NVFP4 expert object, split by Task 5, with shared experts and the router left in BF16.

**Files:**

- Modify: `src/tensorfold/families/glm5_next/cuda/nvfp4_load.py`
- Modify: `src/tensorfold/families/glm5_next/cuda/weights.py` (the MoE construction around `moe_exl3` / the MLX expert stack)
- Test: `tests/cuda/test_glm5_nvfp4_loader.py`

**Step 1: Read the expert constructor**

Read `src/tensorfold/cuda/nvfp4/experts.py` and the Flash Next caller (`src/tensorfold/families/qwen4_exp/cuda/nvfp4_moe.py` if that is who builds the `[E, N, K]` tensor). Write down the exact constructor arguments in the test docstring. Do not guess the layout from this plan.

**Step 2: Write the failing test**

Synthetic experts: `E=4` (not 288), `N` and `K` multiples of 16, ModelOpt suffixes from Task 1. Assert:

- The routed-expert object is the existing NVFP4 expert type, not `Exl3Experts` and not `make_q4`.
- Shared-expert and `mlp.gate` tensors in the same fixture come back BF16.
- A group-16-illegal `N` raises from Task 5's splitter rather than packing a partial group.

**Step 3: Run to see it fail**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_loader.py -q -k experts
```

Expected: FAIL — `routed experts are not wired`, from Task 4.

**Step 4: Wire the constructor**

Call the existing expert builder. Do not add a GLM-specific GEMM. If the constructor cannot take a GLM name prefix without a Qwen-specific string check inside `experts.py`, fix that check so it is shape-based, and add a one-line comment that Qwen and GLM share it. Do not copy the kernel.

**Step 5: Run**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_loader.py tests/cuda/test_nvfp4_experts.py -q
```

Expected: PASS, or GPU-skipped with the same skip mark as the existing expert tests. A skip is not a qualification. Say so in the commit message if no GPU was present.

**Step 6: Commit**

```bash
git add src/tensorfold/families/glm5_next/cuda/nvfp4_load.py \
        src/tensorfold/families/glm5_next/cuda/weights.py \
        tests/cuda/test_glm5_nvfp4_loader.py
git commit -m "feat: load GLM routed experts through the NVFP4 expert kernel"
```

---

### Task 7: Forward calls the NVFP4 modules and keeps everything else

**Objective:** A tiny CUDA forward of one dense layer and one MoE layer runs the NVFP4 modules for the quantized projections and the existing BF16 path for attention, gate, and shared experts.

**Files:**

- Modify: `src/tensorfold/families/glm5_next/cuda/forward.py`
- Modify: `src/tensorfold/families/glm5_next/cuda/weights.py` if the layer objects still cannot hold an `Fp4Linear`
- Test: `tests/cuda/test_glm5_nvfp4_loader.py`

**Step 1: Write the failing test**

On GPU only (`pytest.mark.skipif(not torch.cuda.is_available())`):

- Build a 1-layer fixture with an NVFP4 dense MLP and BF16 attention tensors.
- Run one eager forward row.
- Assert the NVFP4 linear's forward was the one that ran (spy, or compare against `Fp4Linear` applied to the same input outside the model).
- Assert the result is finite.
- A second fixture with a BF16 gate and BF16 shared expert must not call `Fp4Linear` for those names.

This test is exactness against the module, not against vLLM. vLLM comes in Task 9.

**Step 2: Run to see it fail**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_loader.py -q -k forward
```

Expected: FAIL — forward still calls `make_q4` / the MLX matmul.

**Step 3: Wire the call**

Dispatch on the object type the loader returned. Do not add a quant string check scattered through `forward.py` if the object already knows how to multiply. Match `adding-a-cuda-family.md`: one row must be the same bits alone and inside a verify window, once windows exist. For this task, one eager row is enough.

**Step 4: Run**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_loader.py tests/test_glm5_ported_kernels.py -q
```

Expected: PASS or GPU-skipped. Do not mark the port qualified on a skip.

**Step 5: Commit**

```bash
git add src/tensorfold/families/glm5_next/cuda/forward.py \
        src/tensorfold/families/glm5_next/cuda/weights.py \
        tests/cuda/test_glm5_nvfp4_loader.py
git commit -m "feat: run GLM dense and routed NVFP4 projections in forward"
```

---

### Task 8: Startup contract for drafts, vision, and two ranks

**Objective:** The engine refuses the cases this plan does not qualify, and it does not refuse `--tp 2` for this checkpoint.

**Files:**

- Modify: `src/tensorfold/families/glm5_next/cuda/engine.py`
- Modify: `src/tensorfold/families/glm5_next/__init__.py` only if the CLI admits the checkpoint before the engine does
- Test: `tests/cuda/test_glm5_nvfp4_config.py`

**Step 1: Write the failing tests**

```python
def test_glm_nvfp4_refuses_vision():
    from tensorfold.families.glm5_next.cuda.engine import GlmEngine
    with pytest.raises(ValueError, match="vision"):
        GlmEngine("unused", tp=2, rank=0, master="127.0.0.1", vision=True, quant="nvfp4")


def test_glm_nvfp4_does_not_copy_the_qwen_one_gpu_refusal():
    """GLM cannot drop --tp 2. The Qwen message must not be what we raise."""
    from tensorfold.families.glm5_next.cuda import engine as glm_engine
    src = Path(glm_engine.__file__).read_text()
    assert "drop --tp 2" not in src
    assert "run on one GPU" not in src
```

Adjust `GlmEngine` to the class name in `engine.py` (`load` the file; the docstring at the class says "GLM-5.3-Flash on two ranks"). If vision is not an engine argument, assert the CLI path that would pass `--vision` raises before NCCL, and monkeypatch NCCL the way `test_an_nvfp4_checkpoint_refuses_two_ranks` does so a mistaken start fails the test.

Drafts: if Task 1 recorded `has_mtp` false, startup with drafts enabled must raise a message that names `--no-drafts`, before any weight pull. Do not pull `incoai/GLM-5.3-Flash-DFlash2`.

**Step 2: Run to see it fail**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_config.py -q -k 'vision or one_gpu or drafts'
```

Expected: FAIL.

**Step 3: Implement the refusals only**

Do not start NCCL in these tests. A two-rank NVFP4 serve is allowed to proceed past the one-GPU check. It is not allowed to proceed into an unqualified vision path or an unqualified drafter.

**Step 4: Run**

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_config.py tests/cuda/test_qwen27_nvfp4.py tests/cuda/test_flashnext_nvfp4_loader.py -q -k 'two_ranks or vision or ple'
```

Expected: GLM tests pass. Qwen and Flash Next still refuse `--tp 2` on NVFP4. If those tests fail, revert the shared helper. Do not "fix" them by allowing two-rank Qwen NVFP4.

**Step 5: Commit**

```bash
git add src/tensorfold/families/glm5_next/cuda/engine.py tests/cuda/test_glm5_nvfp4_config.py
git commit -m "feat: GLM NVFP4 startup refuses vision and unqualified drafts"
```

---

### Task 9: Logit match against vLLM on the same NVIDIA weights

**Objective:** A short greedy completion matches vLLM on `nvidia/GLM-5.3-Flash-NVFP4`. This is the gate. Speed is not measured in this task.

**Files:**

- Create: `tools/glm_nvfp4_logit_match.py` (a script, not a server)
- Do not change kernel code unless the match fails and the failure is a mapped-tensor bug from Tasks 4–7.

**Step 1: Do not pull weights until this step is approved**

The checkpoint is large. Record the Task 1 byte count. Do not delete an existing serve's weights to free space. Do not stop that serve. This task needs its own approval and an exclusive window only if both engines cannot be resident together. If they cannot, stop and report the byte counts. Do not improvise a park.

**Step 2: Write the comparison script**

Inputs: two base URLs, one prompt, `max_tokens=32`, greedy (`temperature=0`).

```bash
python tools/glm_nvfp4_logit_match.py \
  --left http://127.0.0.1:8000/v1/completions \
  --right http://127.0.0.1:8080/v1/completions \
  --model nvidia/GLM-5.3-Flash-NVFP4 \
  --prompt 'The capital of France is' \
  --max-tokens 32
```

Expected on success: both completion strings equal, token ids equal, exit 0. Expected on failure: exit 1 and the first diverging token index. Do not print a pass if either server errors.

TensorFold serve, only inside the approved window, both ranks, drafts off:

```bash
tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --tp 2 --rank 1 --master RANK0 --no-drafts
tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --tp 2 --rank 0 --master RANK0 --no-drafts --host 127.0.0.1 --port 8080
```

vLLM must be the same checkpoint, greedy, no speculative decoding. Record both version strings in the script's output.

**Step 3: Run one prompt**

One prompt is the gate, not a benchmark. If token 0 diverges, stop. Do not try a longer prompt to "average out" the miss. Inspect KV-cache scheme (Task 1 / the config's `kv_cache_scheme`) before changing a kernel. A bf16 latent cache against a vLLM FP8 KV cache is a known way to fail this gate.

**Step 4: Commit only a passing script and a note of the match**

```bash
git add tools/glm_nvfp4_logit_match.py docs/plans/notes/nvidia-glm-nvfp4-logit.md
git commit -m "test: greedy token match for GLM NVIDIA NVFP4 against vLLM"
```

The note records date, both version strings, the prompt, `max_tokens`, and `MATCH` or `DIVERGE` plus the index. A `DIVERGE` note is still committed if the script is the harness. Do not commit a docs change that says the port is qualified.

---

### Task 10: Same-checkpoint speed, one request, only after MATCH

**Objective:** Measure TensorFold against vLLM on `nvidia/GLM-5.3-Flash-NVFP4`, one request at a time, and write the numbers next to the logit note. Do not compare to the RedHat vLLM serve in this task.

**Files:**

- Modify: `docs/plans/notes/nvidia-glm-nvfp4-logit.md` (append a speed section)
- Modify: `docs/recipes/glm-5.3-flash.md` and the GLM row of `docs/recipes/cuda.md` only if Task 9 was `MATCH`

**Step 1: Refuse to run if the note does not say MATCH**

If Task 9 diverged, stop. A faster wrong server is not the project.

**Step 2: Measure**

Use the public fixture command in `README.md` (measurements section) if it can target this family. If it cannot, time the same greedy completion endpoint Task 9 used, 64 tokens, 5 seeds, and say the command in the note. One request. Both ranks. `--no-drafts` for the first table, so drafting does not confound the loader. A second table with MTP is allowed only if Task 1 found an MTP head and it loaded on the BF16 path.

Record tok/s and time-to-first-token at one short prompt and one prompt of at least 32k, if the startup memory estimate admits 32k. If it does not, record the refusal and the estimate. Do not raise the context by deleting another model's weights.

**Step 3: Update the recipe only with measured cells**

Replace "not read" in the GLM NVFP4 cell with the checkpoint id and "two ranks, logit-matched against vLLM on one greedy prompt, speed table below". Copy the measured numbers. Do not copy the Qwen 1.4–2.0× line into the GLM recipe.

**Step 4: Commit**

```bash
git add docs/recipes/glm-5.3-flash.md docs/recipes/cuda.md docs/plans/notes/nvidia-glm-nvfp4-logit.md
git commit -m "docs: record GLM NVIDIA NVFP4 logit match and one-request speed"
```

---

## After this plan, not in it

RedHat compressed-tensors (`RedHatAI/GLM-5.3-Flash-NVFP4`) is a second plan. Start it only after Task 9 is `MATCH` on the NVIDIA export. The Qwen loader's reciprocal `weight_global_scale` branch is the thing to reuse. Do not reopen EXL3.

A multi-request serve is a third plan. Upstream GLM CUDA serves one request at a time. Beating vLLM on one stream can still lose once more than one request is in flight. Do not claim a replacement for the existing vLLM serve from Task 10's table.

---

## Verification checklist

- [ ] Base is `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`, or this plan's path citations were updated after a rebase.
- [ ] Task 1 census is in `docs/plans/notes/nvidia-glm-nvfp4-index.md` and `mlp.experts` is non-zero.
- [ ] `python -m pytest tests/cuda/test_glm5_nvfp4_config.py tests/cuda/test_glm5_nvfp4_loader.py tests/cuda/test_glm5_nvfp4_split.py -q` passes, with GPU skips called out rather than treated as qualification.
- [ ] Qwen and Flash Next still refuse `--tp 2` on NVFP4.
- [ ] No `incoai` drafter was pulled.
- [ ] No existing vLLM serve was stopped or had its weights deleted.
- [ ] Task 9 note says `MATCH` or `DIVERGE`. Speed docs exist only after `MATCH`.
- [ ] Recipe text does not say GLM NVFP4 is qualified for vision, DFlash2, concurrency, or long context unless a test recorded that.
