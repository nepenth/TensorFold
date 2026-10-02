# GLM-5.3-Flash NVIDIA NVFP4 Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task.
>
> **This commit is the plan only.** Do not implement NVFP4, do not pull weight shards, and do not start a serve in this commit.

**Goal:** Teach TensorFold's `glm5_next` CUDA path to load and serve `nvidia/GLM-5.3-Flash-NVFP4` in that checkpoint's math (NVFP4 weights, FP4 activations under the stored global input scale, per-16 runtime block scales, FP4×FP4 on SM 12.x), on two ranks, with CUDA graphs that follow each step's routing. Qualify it against a recorded vLLM reference on the same weights. Publish speed only after that qualification, and only against the traffic model below, which is a partial bound.

**Architecture:** Reuse `format.scheme`, `Fp4Linear.from_checkpoint`, `checkpoint.quant4`, and the MMA in `lane4.cu` / `gemm_ck.cu`. Do not write a new multiply, do not vendor CUTLASS or FlashInfer, and do not send routed experts through `experts.cu` (bf16 activations).

`checkpoint.matmul_group` quantizes **one** input and applies every matrix to that same quantized tensor. That is the dense gate/up case, and the expert gate/up case, because those projections share the residual. It is not the down case. Each expert's SwiGLU output is a different vector. A shared `down.input_scale` does not make those vectors one tensor.

The serve path, including CUDA graphs, indexes expert weights from a device table built at load. `graphs.Graphs` captures `forward.compute`. A Python loop over `Fp4Linear` objects bakes those pointers into the graph; writing new ids into a buffer afterward does not retarget them. Recapturing after every route, and copying the selected experts' weights into a fixed buffer each token, are both rejected: the first rebuilds the graph inside the forward, the second spends the bandwidth the quant was meant to save.

Decode and prefill share the numeric contract and the weight layout. They do not share a scheduler. Decode has eight experts for one row. A prefill chunk has a different route on every row.

**Tech stack:** TensorFold 0.6.2 (Apache-2.0), base `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`. PyTorch CUDA. SM 12.x block-scaled FP4 MMA. Upstream parent: [ashhart/TensorFold](https://github.com/ashhart/TensorFold).

**Pin:** Implement against `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`. Do not rebase onto a newer `main` as part of these tasks. An upstream move is a separate requalification: re-read every path this plan cites, then edit. Task 0 records the commit actually checked out.

---

## Decision record

Quality bar, in order:

1. `nvidia/GLM-5.3-Flash-NVFP4` — official ModelOpt W4A4, recipe name `nvfp4_experts_dense_mlp-kv_fp8_cast`, producer `modelopt 0.47.0.dev393+ga4bc45b30.d20260828`. **This plan.**
2. `RedHatAI/GLM-5.3-Flash-NVFP4` — compressed-tensors. **Not this plan.**
3. EXL3 / TR3 4bpw — **out of scope.**

The Hugging Face card's sentence "only sparse MoE shared experts and dense MLP are quantized" does not match the file. The recipe name and the index do: routed experts and dense MLPs carry scales; shared experts, attention, the router, embeddings, `lm_head`, and layer 45 do not. Trust the index.

Do not stop or unload the existing multi-request vLLM serve for this plan, for CPU tests, or to pull shards. Two copies of this model do not fit on the two Sparks (see Capacity). The comparison is a recorded reference, then a later TensorFold run. A single-stream win does not replace a server that already runs more than one request.

### What an outside review changed

Checked against the pin on 2026-10-02. Accepted:

- Downs need their own rows. `matmul_group` cannot run them.
- Device-indexed expert dispatch is required before any graph is called qualified. A device id buffer alone does not redirect a captured weight pointer.
- Prefill packs token–expert pairs, runs the expert, and scatters back. It does not push every row through every expert that appears in the chunk.
- `--no-drafts` does not skip MTP. `MTP_DEFAULT` is `"1"`. `mtp_head` honors `serial_only` only when `TF_GLM_MTP=auto`.
- Bitwise equality is the contract when two paths use the same kernels and the same reduction order. Split-versus-unsplit and TensorFold-versus-vLLM are numerical comparisons with a written dtype and order, not bitwise tests. `qmm.split_k` depends on `n` and `k`, so a TP split can change the reduction.
- `matmul` copies a rejected output back; `matmul_group` does not. `_out` drops a buffer that is noncontiguous or the wrong dtype.
- Column-split legality is on the original packed width: `P % 64 == 0` before the split, so each rank's logical K (`P`) is a multiple of 64. `P % 32 == 0` lets `P = 96` through and then fails `Fp4Linear`.
- The old ~80 tok/s figure is a MoE-only partial bound. BF16 KDA Q/K/V/O is more traffic than the routed experts. It is still not a whole-model ceiling.
- One model copy per pair of Sparks. The harness records the reference and compares later.
- The France prompt is a smoke test. Qualification compares teacher-forced next-token distributions, not retokenized strings.
- A 32k timing run needs an explicit `--context`. The default window stays on the dense path.

Narrowed, not rejected:

- 31 tok/s is the streaming arithmetic for the partial table (experts + shared + dense MLP + head + router + KDA Q/K/V/O). It still omits the 11 sparse-attention projections, mHC, cache, and collectives. Do not treat 31 as the number the server should hit.
- CPU loader and split work does not wait on a live vLLM process. The recorded reference is required before any serve comparison, and collecting it still needs the approval this plan already requires. It does not authorize stopping the current server.
- FlashInfer issue 2723 is a historical SM120 grouped-GEMM failure, since closed. It is not a proof that every current CUTLASS build is wrong, and it is not a reason to vendor CUTLASS. The MMA stays `lane4.cu`.
- "Four Over Six" (arXiv:2512.02010) changes the quantization recipe. It is not a way to match this checkpoint.

---

## What the checkpoint actually stores

Measured 2026-10-02 from the public index and from safetensors headers plus the leading 8 KiB of `input_scale` values in `model-00001-of-00033.safetensors`. No shard body was pulled. Task 1 re-checks these facts against a pinned revision, not floating `main`.

Official export: https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4

- 147,661 tensors, 33 shards, `metadata.total_size` = 204,419,110,596 bytes (190.4 GiB).
- `quant_method` `modelopt`, `quant_algo` `NVFP4`. Weights and the **global** activation factor are static. Group size 16. The per-16 e4m3 block scales of an activation are computed at runtime by `quant4` from the values in that block. They are not stored.
- `kv_cache_scheme`: static FP8 (`num_bits` 8, `type` float). No KV scale tensor is in the index.
- Text config: 45 layers, of which 34 are `linear_attention` and 11 are `deepseek_sparse_attention`. 3 dense MLP layers, then MoE. Hidden 4096, dense intermediate 12288, MoE intermediate 2048, 288 routed experts, 1 shared, top-8, `routed_scaling_factor` 2.5, `norm_topk_prob` true, scoring `sigmoid`, `moe_router_dtype` `float32`, `swiglu_limit` 10. Linear attention: 64 heads, head dim 128. `qk_rope_head_dim` 0, `kv_lora_rank` 512, `index_topk` 2048, vocab 154880, context 1,048,576, `num_nextn_predict_layers` 1.

Names. There is no `weight_packed` and no `weight_global_scale`. Codes are `.weight` dtype `U8`. `format.scheme` returns `nvfp4` for `U8` weight plus `F8_E4M3` `weight_scale`.

| Tensor | Stored shape | Logical GEMM |
| --- | --- | --- |
| Expert `gate_proj` / `up_proj` `.weight` | `U8 [2048, 2048]` | `[2048, 4096]`, low nibble first |
| Expert `gate_proj` `.weight_scale` | `F8_E4M3 [2048, 256]` | one e4m3 per 16 K |
| Expert `down_proj` `.weight` | `U8 [4096, 1024]` | `[4096, 2048]` |
| Expert `down_proj` `.weight_scale` | `F8_E4M3 [4096, 128]` | |
| Dense layer-0 `gate_proj` `.weight` | `U8 [12288, 2048]` | `[12288, 4096]` |
| Dense layer-0 `down_proj` `.weight` | `U8 [4096, 6144]` | `[4096, 12288]` |
| `weight_scale_2`, `input_scale` | `F32 []` | per projection, replicate on both ranks |
| Shared expert `gate_proj` | `BF16 [2048, 4096]` | not NVFP4 |
| Router `mlp.gate` | `BF16 [288, 4096]` in the file | config asks for fp32 scores |
| Layer 45 expert `gate_proj` | `BF16 [2048, 4096]` | 889 tensors, none scaled |

`weight_scale_2` may differ per expert and between gate and up. Sharing `input_scale` does not license sharing the weight scale. SGLang issue 21802 is a different model in which a fused gate/up kept one global weight scale and dropped the other. This loader keeps one `Fp4Linear.scale` per projection.

Shard 1 activation scales (2,092 scalars, 39 MoE layers, 18 experts each): one gate scale per layer, `up_proj` bitwise equal (697/697), one distinct down scale per layer. Examples: layer 3 gate = up = `0.001139323`, down = `0.037202381`; layer 44 gate = up = `0.007905507`. The loader asserts this on every shard and stops if a shard disagrees. That assertion allows one quant of the residual for every selected gate and up. It does not allow one quant of a single intermediate for every down.

ModelOpt's global weight scale is passed through as stored. The reciprocal is the compressed-tensors path in `qwen3_5/cuda/nvfp4_load.py`. The global activation factor is `input_scale`, exported by ModelOpt as `amax / (6 * 448)`.

### Routed dataflow

For `R` tokens on one rank:

| Object | Shape |
| --- | --- |
| Residual | `[R, 4096]` |
| Expert ids | `[R, 8]`, plus the shared expert |
| Gate and up outputs | `[R, 8, 1024]` after the TP row split |
| Clamped SwiGLU | `[R, 8, 1024]`, each row its own vector |
| Down partials | `[R, 8, 4096]` fp32, then the weighted sum into `b.part` |

Gate and up: quantize `x` once under the layer gate scale, then each selected expert's own weights. Down: quantize the `R * 8` intermediate rows under the layer down scale (one launch is fine), then expert `e` reads only the rows that routed to `e`.

Prefill, for a 2,048-row chunk: `2048 * 8 = 16384` assignments. Spread across 288 experts that is about 57 rows per expert when routing is even, not 2,048. The order is: assignments, rows packed per expert, gate/up GEMM, quantize those packed rows, down GEMM, scatter back to `[R, 8, hidden]`, then the existing weighted combine. An empty expert launches nothing. A chunk where every row picks different experts still scatters correctly.

---

## What upstream already does

Fork parent: https://github.com/ashhart/TensorFold/commit/56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21

`weights.load` rejects anything other than `mlx` and `exl3`. `Config.group_size` defaults to 64 when the key is absent; NVFP4's 16 lives in `config_groups`. Do not read `cfg.group_size` as 16. `bits_of` defaults a missing `bits` to 4, which this config survives.

`QUANT_METHODS["cuda"]` is `("mlx", "exl3")`. `require_readable` rejects `modelopt` first. `check()` then treats it as a bad MLX `(4, 64)`. `tests/cuda/test_glm_split_and_policy.py` asserts that tuple. All three change together. Do not add `compressed-tensors`.

`MTP_DEFAULT = "1"` in `cuda/engine.py`. `mtp_head` returns true for `"1"` even when `serial_only` is true. `"auto"` is the setting that turns the head off for `--no-drafts`. Unset, `--no-drafts` still loads layer 45. `tests/test_glm_mtp_setting.py` locks this. NVFP4 must not depend on the operator exporting `TF_GLM_MTP`. This quant refuses to load layer 45. `TF_GLM_MTP=1` on an NVFP4 checkpoint raises before NCCL, naming the head as unqualified. `0` and `auto` with `--no-drafts` load no layer-45 tensor and capture no MTP graph. Drafts disabled and MTP weights absent are both required; the flag alone is only the first.

The EXL3 arm builds `draft_head = quantize4(head.weight)` even when `mtp` is false. The NVFP4 serial path does not. That copy is for draft steps.

GLM CUDA is two ranks. `GlmEngine` constructs NCCL before the MTP check, captures graphs for rows 1..6 (`GRAPH_ROWS` includes 5 and 6 in the engine; `Graphs` defaults to 1..4 — follow the engine), and all-gathers fp32 `b.part` inside the capture. `gather` stacks rank-ordered partials; `hc_post` sums them rank 0 first. About two gathers per layer, about 90 per token, each one-row partial 16 KiB (`4096 * 4`). Latency of that many collectives is part of the profile, separate from GEMM time. Do not change the reduction dtype to make them faster.

`serve_options.check` already raises `GLM-5.3-Flash image input is currently MLX-only` before the engine. `tests/test_vision_glm_config.py` locks that string. Do not add a second vision check, and do not construct `GlmEngine` in a CPU test (it calls `set_device` and NCCL immediately).

Without `--context`, the window is `cfg.dense_limit` and attention stays dense (`engine.py`). A 32k prompt on the default launch is not a long-context run. `index_topk` is 2048, so the dense limit is `index_topk + index_kpool - 1`.

EXL3 is the BF16 pattern for attention, norms, router storage, shared expert, embed, and `lm_head`: `make_b16` on `.weight`. The router multiplies bf16 operands and writes fp32 (`glue.router`). `glue.select` applies sigmoid, adds `e_score_correction_bias` for the choice, breaks ties toward the lower id, and normalizes the unbiased scores when `norm_topk_prob` is set, then multiplies by `routed_scaling_factor`. The shared expert is appended at weight 1. The config's `moe_router_dtype: float32` means the comparison against vLLM has to look at scores, not assume the bf16 matmul matches. Both ranks must pick the same ids before the partials are combined.

`glue.swiglu`: `min(gate, 10)`, `clip(up, -10, 10)`, then `bf16(bf16(silu(gate)) * up)` with `up` still fp32 for the multiply. `checkpoint.mlp_prompt` runs SwiGLU epilogue 2 (`SWIGLU_FP32`), which does not clamp and keeps the product in fp32. Epilogue 1 rounds `up` to bf16 first, which is also not `glue.swiglu`. Fusion stays off.

`checkpoint.matmul` re-quantizes, then copies into `out` when `_out` refused the buffer. `matmul_group` reuses one quant and does not copy back. `_out` accepts a buffer only when it is contiguous and the requested dtype. A multi-row slice of a stacked gate/up buffer can fail that test while a one-row view passes. The scheduler either rejects a bad buffer or copies back. It does not return a tensor the caller did not pass.

`Buffers.ey` is bf16 on the prefill path. Down partials that will be reduced live in fp32. `glue.combine` already sums slot-major `y` in fp32 and notes that prefill `y` may be bf16; the NVFP4 path passes fp32 slot outputs so the combine does not round first.

`experts.cu` is bf16 times dequantized FP4. Tests lock that. Flash Next calls it. This plan does not.

Qwen and Flash Next refuse NVFP4 `--tp 2` (`one GPU`). Do not edit those tests.

`split.py` has no `F8_E4M3`. `capacity.SIZES` and `direct_read.py` do. `rule()` raises if a name matches more than one class, so a scalar must return `"rep"` before the row/column scan. `RankReader` splits on the CPU (`split_bytes`) and on a CUDA prefetch (`split_device`), and a pre-split rank folder is a third path. Task 4 covers all three.

`qmm.split_k(n, k)` depends only on shape. Halving `n` or `k` can change the slice count. Task 9 does not require bitwise equality between a split pair and an unsplit GEMM.

`mla_geometry` stores the latent as `count * capacity * kv_lora_rank * 2` bytes on the rank, not divided by world. Eleven sparse layers, 32,768 tokens, width 512: about 352 MiB bf16 per rank, about 176 MiB if the latent were fp8. A 2,048-row chunk of fp32 down slots is `2048 * 8 * 4096 * 4 = 256 MiB` per rank before gate/up workspace. At 32k the expert scratch can exceed the fp8-latent saving. Account for both.

`Weights.nbytes` does not walk an `Fp4Linear`. Admission uses `split_weights(rule)` plus `mla_geometry`. NVFP4 scratch has to be added there the way EXL3 scratch already is. A post-load snapshot is not a prefill budget.

---

## Traffic model

These are streaming calculations: one decoded token, one DRAM read of each listed matrix, TP=2, 273 GB/s from the DGX Spark hardware guide. They are not measured traffic and not a throughput target. NVIDIA's 273 GB/s is a spec, not a sustained rate.

Per routed expert, codes plus block scales: `3 * (2048*2048 + 2048*256) = 14,155,776` bytes. Top-8 across 42 MoE layers, half a tensor per rank: **2.378 GB/token/rank**. All 288 experts resident, not per token: `42 * 288 * 14,155,776 ≈ 171.2 GB` per complete copy. Two copies are about 342 GB, and two Sparks have 256 GB of unified memory between them. A second copy on CPU on those nodes is the same memory. The reference and TensorFold do not run together.

| Component | GB/token/rank |
| --- | ---: |
| Top-8 routed experts, 42 layers, NVFP4 | 2.378 |
| BF16 shared experts, 42 layers | 1.057 |
| Three dense NVFP4 MLPs | 0.127 |
| BF16 `lm_head` (vocab/2) | 0.634 |
| BF16 routers | 0.099 |
| BF16 KDA Q, K, V, O, 34 layers | 4.563 |
| Partial sum | 8.859 |

KDA: `34 * 4 * 4096 * (64 * 128) * 2 / 2 = 4.563 GB`. `273 / 8.859 ≈ 31` tokens/s if nothing else moved and every kernel hit the spec bandwidth. The 11 sparse-attention projections, the extra KDA projections (`f_a`, `g_a`, `b`, conv), mHC, the latent, and ~90 collectives are not in the table. The old 3.4 GB / ~80 tok/s figure is the MoE-plus-shared subtotal only. Use it when profiling the expert kernel. Do not divide a full-token measurement by 3.4 GB and call the result efficiency.

Layer 45, if it were loaded: 288 BF16 experts, three `[2048, 4096]` matrices, half per rank, about **6.75 GiB/rank** before the rest of that layer. This plan does not load it.

A decode that launches `42 * 8 * 3 = 1008` expert GEMMs per token pays launch overhead on top of the bytes. Quantizing the residual once saves a few kilobytes of activation traffic. The bytes that matter are the KDA projections and the expert weights. Order of work: correct device-indexed scheduler and the packed-prefill quantizer, then measure launch gaps and achieved bandwidth on the TP-local shapes, and in the same profile time the BF16 KDA GEMMs, the head, and the collectives. A clamped SwiGLU fusion is allowed only after it matches `glue.swiglu` on the same rows, including the bf16 rounding of the sigmoid. Do not retune `split_k` without rerunning the numerical check. Do not assume a Qwen tile choice is right for `N` in `{1024, 2048, 4096, 6144}`.

Same-checkpoint TensorFold numbers that are not this model: Qwen3.8-27B NVFP4 decode 1.4–2.0× vLLM on one RTX PRO 6000 (`CHANGELOG` 0.6.1); Flash Next NVFP4 1.13–1.52× on one Spark, W4A16 experts (`CHANGELOG` 0.3.6.3). Do not copy either ratio into the GLM recipe. The MLX GLM table in `docs/recipes/glm-5.3-flash.md` is the wrong quant.

The model card's vLLM command is TP 4 with expert parallel on GB200-class hardware, image tag `glm53-flash-arm64-cu130`. The tag names an architecture and a CUDA build. It does not by itself prove SM 12.1. The speed denominator is vLLM `--tp 2` on the same two machines, recorded, not a four-GPU number. vLLM 0.22.1's NVFP4 clamp allow-list is `FLASHINFER_TRTLLM` while the error text also names CUTLASS; later `main` is wider and warns that shape-specific fallbacks still happen. Record the backend the process actually ran, for prefill and for decode. A startup line is not proof the activations stayed FP4.

---

## KV cache and context

Do not invent a KV scale. The scheme says static FP8 and the index has no scale tensor. TensorFold's latent is bf16, width 512, rope dimension 0, replicated on each rank.

The comparison manifest records the reference cache dtype, layout, and whether `--kv-cache-dtype bfloat16` starts. If both sides can run bf16, the first numerical compare uses bf16 so a bad GEMM is not mixed with a cache cast. If the reference is fp8 only, stop and write down the writer (scale, layout, whether the 512-d latent is what is stored) before adding a cast. Matching the dtype string is not enough: latent layout, KDA conv and recurrent state, indexer pools, and prefix reuse have to be named. A September 2026 study of GLM-5.3-Flash hybrid-state restore (arXiv:2609.15030) found a full-hit mismatch where restored state covered the prompt and the scheduler credited one fewer token. That run was RedHat NVFP4 at TP 4. It is a failure mode to test, not a result for this port.

Tests, once a serve exists: cold versus prefix reuse, a continuation after an unrelated request, reset, and lengths around the dense limit and the prefill chunk. Short dense completions do not qualify the sparse path.

FP8 latent is in scope for the long-context configuration after the writer is known. It is not a substitute for the 256 MiB expert-slot scratch.

---

## Numerical contracts

Three different comparisons:

| Contract | Requirement |
| --- | --- |
| Load and pack | Exact codes, scales, shapes, and which scalar is `weight_scale_2` versus `input_scale` |
| Scheduler versus the eager per-expert oracle, same order | Bitwise on the fp32 partial |
| TP split versus unsplit, or TensorFold versus vLLM | Written dtypes and reduction order, plus max absolute error, RMS, and top-2 margin. Not bitwise |

`split_k` may differ after a split. Say so in the Task 9 note. A token change with a top-2 margin below twice the logit error is a near-tie, recorded as such. It still fails a token-parity claim. It is not by itself a wrong load. The plan stops on an unexplained divergence; the harness has to show which layer moved first.

Declared NVFP4 order: gate/up bf16 out of the MMA (or fp32 if a test asks), `glue.swiglu` as written, down accumulation fp32, combine in slot order with the shared expert last at weight 1, TP sum of fp32 partials rank 0 first.

Quantizer tests, against a small independent reference (the ModelOpt scale and round rules, or a few dozen lines that implement them, not a second call to `quant4`): zeros, values under the smallest e4m3 scale, midpoint ties, saturation at 6, non-unit global scales, and values around the SwiGLU clamp. "Diagnosing FP4 inference" (arXiv:2603.08747, Qwen2.5, not GLM) found up and down projections sensitive, including early layers. Probes on a divergence start at layers 0–2 and at the first MoE layer's up / SwiGLU / down, not only at the last layer.

---

## Non-goals

- No new MMA. A device-indexed launch that calls `lane4` / `gemm_ck` is in scope. A new numeric recipe is not.
- No CUTLASS, FlashInfer, or TensorRT-LLM dependency.
- No `experts.cu` for this checkpoint.
- No EXL3 work, no RedHat loader, no Apple NVFP4.
- No vision serve.
- No DFlash2 pull, and no layer-45 load. `incoai/GLM-5.3-Flash-DFlash2` is CC BY-NC-ND 4.0.
- No expert parallelism and no `--parallel`.
- No second resident copy, no shard pull, and no exclusive window until the task says so.
- No MLX or EXL3 TensorFold serve in the comparison.
- No "Four Over Six" rescale of this checkpoint.

---

## Stop conditions

- The pinned index disagrees with the census table, or a shard's `input_scale` is not layer-wide for gate=up and for down.
- A split cuts a group of 16, leaves a rank with logical K not a multiple of 64, or splits a scalar.
- The device scheduler does not match the eager oracle bitwise on the fp32 partial, including a second routing and a chunk whose rows disagree.
- The only graph strategy left is recapture-per-token or copying selected expert weights into the capture buffer.
- Teacher-forced token distributions diverge and the layer trace does not explain it. Do not lengthen the prompt to average a token-0 miss.
- Continuing requires stopping the existing vLLM serve before the record/compare harness exists.
- The work needs an MMA other than `lane4.cu` / `gemm_ck.cu` / `gemm_ws.cu`.

---

## Files

Create:

- `docs/plans/notes/nvidia-glm-nvfp4-manifest.md` — commits, checkpoint revision, config hash, reference command once known
- `docs/plans/notes/nvidia-glm-nvfp4-index.md`
- `src/tensorfold/families/glm5_next/cuda/nvfp4_load.py`
- `src/tensorfold/families/glm5_next/cuda/nvfp4_moe.py` — eager oracle, device table, prefill pack/scatter
- `tests/cuda/test_glm5_nvfp4_config.py` (CPU)
- `tests/cuda/test_glm5_nvfp4_split.py` (CPU, plus the CUDA prefetch test)
- `tests/cuda/test_glm5_nvfp4_loader.py`
- `tools/glm_nvfp4_record.py` and `tools/glm_nvfp4_compare.py`

Modify only as the tasks say:

- `src/tensorfold/families/glm5_next/__init__.py` — `modelopt` in `QUANT_METHODS["cuda"]` and in `check()`
- `cuda/weights.py` — admit `modelopt`; BF16 arms for attention and shared expert; no `draft_head` on this path; prefetch the NVFP4 suffixes
- `cuda/split.py` — `F8_E4M3`; scalars return `rep` first; logical-K check
- `cuda/engine.py` — NVFP4 does not load MTP; `TF_GLM_MTP=1` refuses
- `cuda/forward.py` — dense `Fp4Linear`; MoE through `nvfp4_moe`; fp32 slot outputs
- `cuda/geometry.py` — NVFP4 workspace in `mla_geometry` when the quant is modelopt
- `src/tensorfold/cuda/nvfp4/checkpoint.py` — output-buffer contract only if the scheduler uses `matmul_group` for same-input projections. Do not stretch `matmul_group` to many inputs.
- Recipes only after a qualified run

Do not weaken `test_qwen27_nvfp4.py`, `test_flashnext_nvfp4_loader.py`, `test_quant_family_formats.py`, `test_exl3_format.py`, `test_glm_exl3_bits.py`, `test_glm_mtp_setting.py` (the default stays `"1"` for MLX and EXL3), or `test_vision_glm_config.py`.

---

### Task 0: Freeze the manifest

**Objective:** A note names the exact tree and the exact checkpoint revision this work is about. No rebase, no shard pull, no serve change.

Write `docs/plans/notes/nvidia-glm-nvfp4-manifest.md` with:

- `git rev-parse HEAD` and whether `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21` is an ancestor. If it is not, stop. Do not rebase in this task.
- The Hugging Face revision for `nvidia/GLM-5.3-Flash-NVFP4` (the commit the API returns for `main` on this day), the sha256 of `config.json`, and `total_size`. Fetch with a revision URL once you have the commit, not a second floating `main` later.
- A line that the reference server has not been inspected yet, so kernel, cache, and clamp fields are blank until Task 12.

```bash
git add docs/plans/notes/nvidia-glm-nvfp4-manifest.md
git commit -m "docs: pin the GLM NVFP4 tree and checkpoint revision"
```

---

### Task 1: Confirm the census against that revision

**Objective:** The index at the manifest's revision still matches the table. No loader code.

Re-fetch `model.safetensors.index.json` from that revision. Confirm 147661 tensors, 33 shards, `total_size` 204419110596, `weight_packed` 0, `weight_global_scale` 0, layer 45 present and unscaled. Range-read the header of `model-00001-of-00033.safetensors` and confirm expert 0 gate is `U8 [2048, 2048]`, scale `F8_E4M3 [2048, 256]`, `input_scale` `F32 []`.

If a count moved, stop and update this plan before Task 2. Paste the counts into `docs/plans/notes/nvidia-glm-nvfp4-index.md`.

```bash
git commit -m "docs: census nvidia GLM-5.3-Flash NVFP4 tensors"
```

---

### Task 2: Lock today's refusals

**Objective:** CPU tests fail closed on ModelOpt at `require_readable`, `check()`, and `load`.

The fixture is a complete `Config.read` document (`layer_types`, vocab, norms, LoRA ranks, expert fields, `routed_scaling_factor`, `eos_token_id`) plus `quantization_config` (that key, not `quantization`: `config_block` only reads `quantization_config`). Quant block: `quant_method` `modelopt`, `quant_algo` `NVFP4`, group 16, static float4 weights and activations, the producer version above.

Tests: `scheme` on `.weight` `U8` plus `weight_scale` `F8_E4M3` is `nvfp4`. `require_readable` raises. `check()` raises. `load(..., rank=0)` raises `MLX 4-bit or EXL3` once the fixture reaches that line. `rank` is keyword-only.

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_config.py -q
git commit -m "test: lock GLM CUDA refusal of ModelOpt NVFP4"
```

---

### Task 3: Admit the recipe, not only the method name

**Objective:** `modelopt` passes the family gate only when the block is static NVFP4, group 16, float weights and activations. `load` says the tensors are not wired. `compressed-tensors` still dies.

`QUANT_METHODS["cuda"]` gains `"modelopt"` only. `check()` accepts `quant_algo == "NVFP4"` with group 16 and `dynamic: false` on both weights and activations. Anything else raises a message that names the missing field. Do not send it through the MLX `(4, 64)` test.

Update `tests/cuda/test_glm_split_and_policy.py` so the tuple includes `modelopt`. A compressed-tensors fixture still raises. `load` of the good fixture raises `NVFP4 tensors are not wired`.

```bash
python -m pytest tests/cuda/test_glm5_nvfp4_config.py tests/test_glm5_next_family.py tests/cuda/test_glm_split_and_policy.py tests/test_quant_family_formats.py -q
git commit -m "feat: admit the NVIDIA NVFP4 recipe in the GLM family gate"
```

---

### Task 4: Split codes, scales, and scalars

**Objective:** `rule`, `split_bytes`, and `split_device` agree on this export. A pre-split rank folder round-trips.

Return `"rep"` for `input_scale` and `weight_scale_2` before the row/column scan. `.weight` and `.weight_scale` keep today's row rule for gate/up and column rule for down. Add `F8_E4M3` (1 byte, `torch.float8_e4m3fn`) to `DTYPE_BYTES` and `torch_dtype`.

Column split: let `P` be the original packed column count. Logical K is `2P`. After TP=2 each rank's logical K is `P`, so require `P % 64 == 0` on the original tensor. `P % 32 == 0` is the wrong test (`P = 96` yields logical K 96). Do not pad.

Tests:

- Expert gate `U8 [2048, 2048]` row-splits to `[1024, 2048]`; scale `[2048, 256]` to `[1024, 256]`.
- Expert down `U8 [4096, 1024]` column-splits to stored `[4096, 512]` (logical K 1024); scale `[4096, 128]` to `[4096, 64]`.
- Dense down `U8 [4096, 6144]` column-splits to stored `[4096, 3072]` (logical K 6144).
- Scalars `[]` replicate, and `rule` does not call them ambiguous.
- Original packed width 96 raises.
- The same bytes come out of `split_device`.
- Write a one-tensor rank folder with `split_file` and read it back through `RankReader`.

Run this file plus `tests/test_cuda_capacity.py` and `tests/test_cuda_geometry.py`.

```bash
git commit -m "feat: split GLM NVFP4 tensors on groups the kernel can use"
```

---

### Task 5: Quantizer reference, then one dense projection

**Objective:** `quant4` matches an independent scale-and-round on the cases in Numerical contracts. One dense projection becomes an `Fp4Linear` with `act=input_scale` and `scale=weight_scale_2` (not the reciprocal).

The independent reference is a short function in the test, transcribed from ModelOpt's global factor and e4m3-per-16 rule, not a call into `quant4`. Cover the cases listed above.

Loader tests, names from the census (`layers.0.mlp.gate_proj.weight` and the three siblings):

- Missing `weight_scale_2` raises `global scale`.
- `compressed-tensors` raises `compressed-tensors is a later checkpoint`.
- A BF16 shared-expert weight returns `make_b16`.
- `model.visual` is skipped.
- An expert name raises `routed experts are not wired`.
- Gate and up keep distinct `weight_scale_2` values when the fixture gives them different ones.

Point the Task 3 "not wired" assertion at this loader function if `load` now calls it. GPU skip only around the CUDA constructor.

```bash
git commit -m "feat: map one GLM dense MLP projection to Fp4Linear"
```

---

### Task 6: Routed experts load; layer 45 does not

**Objective:** A MoE layer loads `E` experts (the test uses 4) as per-projection `Fp4Linear`s. Shared expert and `mlp.gate` are BF16. Layer-wide `input_scale` is asserted. Serial load does not build `draft_head` and does not read layer 45.

`expert_names` / prefetch currently ask for `.scales` and `.biases`. On this quant prefetch `weight`, `weight_scale`, `weight_scale_2`, and `input_scale` for routed projections, and `.weight` only for the shared expert. Attention uses the BF16 arms. A disagreement in layer-wide scales raises `input scales are not layer-wide`.

`Weights.nbytes` may stay blind to `Fp4Linear`. Do not use it as the residency check. Task 11 adds the geometry term.

```bash
git commit -m "feat: load GLM routed experts as per-projection Fp4Linears"
```

---

### Task 7: NVFP4 startup leaves MTP unloaded

**Objective:** For `quant == "modelopt"`, layer 45 is not requested and no MTP graph is captured. `TF_GLM_MTP=1` raises `unqualified` before NCCL. `0` and `auto` with `--no-drafts` proceed to load. MLX and EXL3 keep today's `mtp_head` behavior, including the default `"1"`.

Test the four settings (`unset`, `0`, `1`, `auto`) against `mtp_head` and against the NVFP4 branch separately, so the existing test that default is `"1"` still passes. Vision stays the `serve_options` error. `engine.py` does not contain `drop --tp 2` or `run on one GPU`.

Qwen and Flash Next `--tp 2` NVFP4 tests still pass.

```bash
git commit -m "feat: GLM NVFP4 refuses the MTP head and keeps two ranks"
```

---

### Task 8: Eager dense and expert oracle

**Objective:** One dense layer and one MoE layer, eager, match a hand-written sequence. No graph yet. Downs are not `matmul_group`.

Dense: gate and up may use `matmul_group` because they share `x` and `act`. Then `glue.swiglu` with limit 10. Then one `checkpoint.matmul` for down under `down.act`, fp32 into the caller's buffer. A row with a gate component above 10 differs from `mlp_prompt`.

MoE oracle, per selected expert, Python loop: same gate/up quant once, per-expert weights, `glue.swiglu`, per-expert quant of that expert's intermediate, per-expert down, fp32 slots, shared expert BF16, router BF16 in / fp32 scores. The test's experts have different weights, so the intermediates differ. A symmetric fixture is not sufficient.

Output contract tests: a noncontiguous multi-row destination is either honored or copied back, and a canary past the written range stays intact. A wrong dtype raises. One row passing does not cover this.

Attention projections in the fixture are `B16` and never enter `Fp4Linear`.

```bash
git commit -m "feat: run GLM NVFP4 eager projections with per-expert downs"
```

---

### Task 9: Two-rank numerical contract

**Objective:** Row-parallel gate/up (concatenate) and column-parallel down (sum fp32 partials in rank order) stay within a declared error of the unsplit projection. Document `split_k` on both shapes. Do not require bitwise equality.

Use the Task 4 shapes and `split_bytes`. Record max abs error. If it is nonzero, the note says whether `split_k` changed. Qwen and Flash Next still refuse `--tp 2`.

```bash
git commit -m "test: GLM NVFP4 two-rank partials stay inside a declared error"
```

---

### Task 10: Device-indexed MoE, then graphs

**Objective:** `forward.compute` does not close over a Python list of selected `Fp4Linear`s. Replay with a new route matches the Task 8 oracle bitwise. Prefill packs and scatters.

At load, stack each layer's expert weights into a device table `[E, ...]` per projection (gate, up, down), with per-expert `weight_scale_2` beside it. The launch reads expert ids from the tensor `glue.select` already wrote. It calls the existing MMA. It does not call `experts.cu`.

1. Quantize the residual once under the layer gate scale.
2. Gate and up for the selected ids, each expert its own weights and its own `weight_scale_2`.
3. `glue.swiglu` with limit 10 on each expert's pair.
4. Quantize the `R * 8` intermediate rows under the down scale. Down for expert `e` reads only its rows.
5. Combine, then `gather`.

Prefill uses the pack → GEMM → scatter order from the dataflow section. Tests: every row a different route, one expert empty, one expert taking most of the chunk, and a second call with a different route. The implementation must not run a row through an expert it did not pick.

Graphs: capture rows 1 and 4. Replay. Change the ids on the device and replay again without recapturing. The second output matches the oracle for the new ids. If the only way to do that is a new multiply, stop and leave the eager oracle in place; do not qualify the graph.

```bash
git commit -m "feat: index GLM NVFP4 experts from a device table"
```

---

### Task 11: Count the bytes the new path allocates

**Objective:** `mla_geometry` grows by the NVFP4 slot scratch for the decode window and for a prefill chunk, and the startup log can separate weight bytes, graph pool, cache, prefill scratch, and headroom.

Include the 256 MiB figure for a 2048-row fp32 down-slot buffer as a comment next to the term, so a later change that materializes `[R, 8, hidden]` shows up in admission. Do not build `draft_head`. Run `tests/test_cuda_geometry.py` and `tests/test_cuda_capacity.py`.

```bash
git commit -m "feat: account for GLM NVFP4 expert scratch in admission"
```

---

### Task 12: Record the reference, then compare

**Objective:** A harness with two modes. Record does not require TensorFold weights resident. Compare does not require vLLM resident. Neither mode stops the existing server unless a separate approval says so, and that approval includes how the server is restored.

**Record** (`tools/glm_nvfp4_record.py`), when a reference is available without taking the machine:

- Manifest fields: container digest or version strings, GPU capability, CUDA, NCCL, FlashInfer or CUTLASS if the log prints them, the MoE backend line for prefill and for decode, whether activations stayed FP4 or fell back, whether the SwiGLU clamp is on, cache dtype and layout.
- Prompt token ids, tokenizer revision, special-token policy, temperature 0.
- Returned token ids from the API field, not from retokenizing text. If the server can return logprobs, store those and label them as logprobs, not as logits.
- One teacher-forced continuation: the same id sequence fed as context, next-token distribution at each position.

If no reference can be observed without exclusive access, the note says that and stops. Do not pull 190 GiB beside a live copy.

**Compare** (`tools/glm_nvfp4_compare.py`): TensorFold against the saved record. Smoke: the France prompt's token ids. Qualification: teacher-forced agreement on the saved continuation, plus max absolute error and top-2 margin where numbers were stored. A string match alone is not `MATCH`.

On the first miss, trace the first layer whose residual differs, and record router ids on both ranks (they must match each other) before looking at the KV writer. Layers 0–2 and the first MoE up/SwiGLU/down are the probes.

TensorFold serve, only inside an approved window, both ranks:

```bash
TF_GLM_MTP=0 tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --tp 2 --rank 1 --master RANK0 --no-drafts
TF_GLM_MTP=0 tensorfold serve nvidia/GLM-5.3-Flash-NVFP4 --tp 2 --rank 0 --master RANK0 --no-drafts --host 127.0.0.1 --port 8080
```

`TF_GLM_MTP=0` is belt and suspenders. Task 7 already refuses to load the head for this quant. The commands omit `--context`, so this is the dense-window qualification. Do not call it 32k.

vLLM, when it is recorded, is `--tensor-parallel-size 2`, greedy, no speculative decoding. Do not copy the card's TP 4.

The note records `MATCH` or `DIVERGE` and the index. A `DIVERGE` note is still committed. Recipe text stays unqualified.

```bash
git commit -m "test: record and compare GLM NVIDIA NVFP4 against a saved vLLM reference"
```

---

### Task 13: Long context is a separate qualification

**Objective:** Do not attach a 32k speed to the Task 12 serve.

A long-context run sets `--context` on both ranks to at least the prompt plus the generated tokens, and it is attempted only after Task 12 is `MATCH` on the dense window. Before any 32k timing, one eager check crosses `dense_limit` and checks the chunk boundary. If that check is absent, the recipe does not mention 32k.

KDA state and the latent are part of the cold-versus-prefix check in the KV section. This task does not add FP8 latent unless Task 12's manifest says the reference was fp8 and the writer is known.

```bash
git commit -m "test: qualify GLM NVFP4 past the dense attention window"
```

---

### Task 14: Profile, then write numbers

**Objective:** One request, two ranks, the device-indexed path, only if Task 12 is `MATCH`. If the serve is still the eager oracle, say so and do not compare it to the 8.9 GB bound.

Repeated identical greedy trials, not five seeds of a deterministic decode. Report generated-token count, time to first token from the stream (not the full non-streaming duration), decode-only tok/s, median and tail inter-token time, warmup excluded, cold prefix and warm prefix separately, context, peak memory, and power if the machine reports it. Separate load and graph capture from the steady state.

Also report, for one decoded token: expert-kernel time, KDA time, head time, and collective time. Put the partial table next to the tok/s. MoE-only efficiency uses 2.38 GB, not 8.86. Full-token efficiency uses 8.86 and still says the 11 sparse layers are missing.

Update `docs/recipes/glm-5.3-flash.md` and the GLM row of `docs/recipes/cuda.md` only to the checkpoint id, "two ranks, teacher-forced match against a recorded vLLM reference, dense window", and the measured table. Do not copy the Qwen ratio. Do not say vision, MTP, concurrency, expert parallel, or long context are qualified unless that task recorded it.

```bash
git commit -m "docs: record GLM NVIDIA NVFP4 match and one-request speed"
```

---

## After this plan

RedHat compressed-tensors, after Task 12 is `MATCH`. Reuse the Qwen reciprocal global scale. Do not reopen EXL3.

MTP on the BF16 layer-45 experts. The tensors exist. This plan refuses them.

A multi-request serve. One stream is not a replacement for the existing vLLM server.

A clamped fusion of SwiGLU into the down quantizer, only with a bitwise match to `glue.swiglu`.

---

## Verification checklist

- [ ] Manifest names the tree and the checkpoint revision. No silent rebase.
- [ ] Census note matches the table, including no `weight_packed` and unscaled layer 45.
- [ ] `pytest` on `test_glm5_nvfp4_config.py`, `test_glm5_nvfp4_loader.py`, and `test_glm5_nvfp4_split.py` passes. GPU skips are not qualification.
- [ ] Qwen and Flash Next still refuse NVFP4 `--tp 2`. MLX/EXL3 `MTP_DEFAULT` is still `"1"`.
- [ ] NVFP4 with `TF_GLM_MTP` unset does not read layer 45.
- [ ] Expert downs are not `matmul_group`. A test with unequal expert weights fails if they were.
- [ ] A graph replay follows a changed device-side route.
- [ ] Prefill does not evaluate unselected experts.
- [ ] No `incoai` drafter and no second model copy. The existing serve was not stopped without an approved restore.
- [ ] Task 12 says `MATCH` or `DIVERGE` on teacher-forced ids, not on a retokenized string. Speed docs exist only after `MATCH`, and they quote the partial traffic table.

---

## References

Format and export:

- NVIDIA, "Introducing NVFP4 for Efficient and Accurate Low-Precision Inference": https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Model card `nvidia/GLM-5.3-Flash-NVFP4` (recipe name, TP4 example, the shared-expert sentence this plan does not follow; test hardware listed as GB200): https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4
- ModelOpt `NVFP4QTensor.get_activation_scaling_factor` (`amax / (6 * 448)`) and block scales: https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
- ModelOpt PTQ recipes: https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md
- OCP Microscaling Formats v1.0 (MXFP4, E8M0, group 32). This export is not that format: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- TensorRT-LLM quantization table (NVFP4 compute and FP8 KV are separate rows): https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html

Engines:

- vLLM `select_nvfp4_moe_backend` at 0.22.1 (clamp list versus error text): https://docs.vllm.ai/en/v0.22.1/api/vllm/model_executor/layers/fused_moe/oracle/nvfp4/
- vLLM oracle on `main` (wider list, shape-specific fallbacks): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
- vLLM ModelOpt loader (NVFP4 with and without quantized activations): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
- vLLM completion protocol (token ids and logprobs are different fields): https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/
- vLLM batch invariance is documented as beta and is not assumed here: https://docs.vllm.ai/en/latest/features/batch_invariance/
- FlashInfer issue 2723 (historical SM120 grouped GEMM failure; do not vendor): https://github.com/flashinfer-ai/flashinfer/issues/2723
- SGLang issue 21802 (fused gate/up dropped a distinct weight scale; different model): https://github.com/sgl-project/sglang/issues/21802
- vLLM issue 53963 (stock SM120 sparse MLA rejects `qk_rope_head_dim` 0): https://github.com/vllm-project/vllm/issues/53963
- vLLM recipe file lists the RedHat NVFP4 checkpoint, not this export: https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3-Flash.yaml

Hardware and research:

- DGX Spark hardware guide (128 GB unified, 273 GB/s): https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- arXiv:2609.15030, hybrid-state cache restore on GLM-5.3-Flash. RedHat NVFP4, TP 4, not this port.
- arXiv:2603.08747, "Diagnosing FP4 inference". Qwen2.5 probe placement, not a GLM measurement.
- arXiv:2512.02010, "Four Over Six". A different quantization recipe. Not used here.

In-tree:

- `src/tensorfold/cuda/nvfp4/checkpoint.py` — `matmul_group` is one input; `_out` and the missing copy-back
- `src/tensorfold/cuda/kernels/qmm.py` — `split_k`
- `src/tensorfold/families/glm5_next/cuda/engine.py` — `MTP_DEFAULT`, `mtp_head`, default dense window
- `src/tensorfold/families/glm5_next/cuda/graphs.py` — capture of `compute`
- `src/tensorfold/families/glm5_next/cuda/glue.py` — `swiglu`, `router`, `select`
- `src/tensorfold/families/glm5_next/cuda/split.py` — `split_bytes` and `split_device`
- `src/tensorfold/cuda/geometry.py` — latent bytes per rank, EXL3 scratch hook
- `src/tensorfold/families/glm5_next/cuda/weights.py` — `draft_head` on the EXL3 arm, prefetch names
