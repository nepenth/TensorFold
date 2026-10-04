# GLM-5.3-Flash NVIDIA NVFP4 Implementation Plan

> **For the orchestrator:** This document is the plan only. Do not implement NVFP4, pull weight shards, or start a serve in a documentation commit. A later implementation uses the work streams in [Work streams](#work-streams). Do not load a second model, stop the existing server, or occupy its GPUs unless a task explicitly says that window was approved, and that approval includes how the server is restored.

**Goal:** Load the pinned NVIDIA ModelOpt checkpoint without changing its stored codes or scale meanings. Run its dense MLPs and routed experts as W4A4 on TensorFold's existing SM 12.x FP4 multiply. Keep the BF16 modules and GLM's existing arithmetic. Serve one request at TP=2, with device-driven routing. Qualify named execution modes against a recorded reference on the same checkpoint.

**Architecture:** Reuse `format.scheme`, `Fp4Linear.from_checkpoint`, `checkpoint.quant4`, and the MMA in `lane4.cu` / `gemm_ck.cu`. Do not write a new multiply. Do not vendor CUTLASS, FlashInfer, or TensorRT-LLM. Do not call `experts.cu` (bf16 activations).

`checkpoint.matmul_group` quantizes one input and applies every matrix to that same tensor. Dense gate/up and expert gate/up may use that, because they share the residual. Expert downs may not. Each selected expert has its own SwiGLU output.

`checkpoint.cpp` `lane` takes one code pointer, one weight pointer, and one scalar. A Python table of experts does not make that binding device-indexed. A small dispatch wrapper may select expert addresses, row destinations, and the existing `mma_fp4` reduction. That wrapper is in scope. A new numeric recipe is not. Do not reuse the lane kernel's split-K grid dimension as an expert dimension.

The first expert storage is a **pointer table**: each `Fp4Linear` owns its packed `words` and `bs` once, and a device table holds addresses, dimensions, and the rounded `act * weight_scale_2`. Do not allocate those tensors and then stack a second copy. Routed-expert codes and scales are about **79.73 GiB per rank** (`42 * 288 * 14,155,776 / 2`). A duplicate would not fit next to the BF16 weights on a 128 GB Spark. An arena that packs straight into one allocation is allowed later only if a profile shows the pointer table missing bandwidth, and only if it still has a single owner.

Prepared activations are not row-major. `lane` checks activation scales as `(K/64, mpad, 4)`. The prompt path can swizzle codes. Gathering "row 0" of the scale tensor does not gather the scales for code row 0. The scheduler either quantizes BF16 rows into an expert-local prepared region, or gathers codes and scales with a layout id. It does not slice a packed buffer and hope.

Decode and prefill share weights and the numeric meaning of a projection. They do not share a bitwise oracle. Decode uses the lane kernel and `split_k`. Prefill uses the prompt GEMM, which is a different reduction. Each is bitwise only against an eager oracle on that same backend.

**Tech stack:** TensorFold 0.6.2 (Apache-2.0), base `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`. PyTorch CUDA. Upstream: [ashhart/TensorFold](https://github.com/ashhart/TensorFold).

**Pin:** Implement against that commit. Do not rebase onto a newer `main` inside these tasks. Record `HEAD`, dirty state, and the checkpoint revision in the manifest. An ancestor check is not enough: uncommitted edits are part of the manifest. A later upstream move is a new manifest and a targeted requalification.

## Where we are

Tracking note: `docs/plans/notes/nvidia-glm-nvfp4-status.md`. Branch `plan/glm-nvidia-nvfp4`. Checkpoint revision `da920bb0b9f4a06727223a349e55468e38352348`. Reference status `BLOCKED_REFERENCE`.

Each expert keeps its own global scales. Rank 0 and rank 1 have each loaded layers 0 through 44, returned one finite bf16 token, and dropped them. They were not resident together. Layer 45 was not loaded. Modelopt does not capture a CUDA graph. No serve.

| Task | State |
| --- | --- |
| 0 Manifest, 0A comparator, 1 census, 2 refusal lock | Done. Reference status remains `BLOCKED_REFERENCE`. |
| 0B Reference | `BLOCKED_REFERENCE`. The previous serve was a different checkpoint. No completion was recorded. It is not the oracle. |
| 3 Admit the recipe | Done. Static NVFP4, group 16 only. Neighbor recipes raise. |
| 4 Splits | CPU path done. `split_device` matches `split_bytes` on one Spark. |
| 5A–5D Codec, prepared rows, one projection | Done as finite bf16 on lane and prompt. Recorded `split_k` is 2. Same-backend eager oracles passed later. No numeric envelope. |
| 11A Inventory | Done. Draft-head byte term is off for this quant. |
| 11B Measure peaks | Partial. Each rank's one-token load peak is measured. Prefill peak and a captured peak are not. Modelopt does not capture a graph. |
| 6 Load | Fixture load, per-expert scales, pack-and-drop of every language-model layer except 45, then a resident load and one token on each rank. After the conv cast, `Weights.nbytes` was 95,140,735,476 bytes (88.607 GiB) on each rank. Each token was finite bf16 `[1, 77440]`. The ranks were not resident together. The tables were dropped. |
| 7 Startup before NCCL | Partial. Explicit MTP, a drafter, and `--parallel` raise before the engine import. The small-tensor two-rank smoke recorded a missing peer and a revision mismatch. That is not a snapshot collective. |
| 8 Eager oracles | Passed on one Spark. Lane and prompt are not compared bitwise. |
| 9A TP partials | Passed on one device. Envelope not frozen. |
| 9B–9C Two ranks | Small-tensor NCCL smoke passed. Not a snapshot load on two ranks. |
| 10A–10C Device dispatch and prefill pack | Passed. 10C is an eager pack, not a grouped device GEMM, and not graph-qualified. Sanitizer zero errors on the canary only. |
| 10D Graph | Primitive graph passed: fixed bank, supplied ids change, table addresses do not. Full-forward graph is not done. The real router is not in that capture. |
| 12 Record and compare, 13 long context, 14 speed | Not started. Reference is `BLOCKED_REFERENCE`. Do not start these from a clear calendar. |

### Spark boundary

The next command is still not `tensorfold serve`.

1. The pinned snapshot is local on both ranks. 33 shards. Index `26765b2601fd246ef361cfb9f5e10f9fb291a59e05ad0a109062f3a4747c7fd1`. Config `41db2811023b40ba4c8f8bbba88bce7dff377af51ecd18b32469a2a07064ebaf`.
2. Header census: one rank keeps 95,777,735,492 bytes (89.200 GiB). The running peak is 99,462,461,744 bytes (92.632 GiB) at layer 43 if the packed layer overlaps the current reader's raw spans. Column spans still include the other rank's bytes. The retained set does not.
3. `nvfp4_admit.guarded` stops in front of each layer unless available bytes cover that layer's increment plus a 16 GiB reserve. It does not read a cgroup cap. The rank-0 load above used that check and was dropped.
4. Each rank returned finite bf16 logits `[1, 77440]` and was dropped. The engine used one prefill row, not 2048. Rank 1 peak allocated was 96,742,671,360 bytes. During that token, 21,244,188 kB remained available. Next proof: both ranks, one real-weight token, then drop. Not a second resident copy on one machine. Not a serve. `spark-llm` is not advertised.

A CPU torch wheel can unskip the `load()` refusal test. It cannot qualify `split_device`, the MMA, graphs, or two-rank collectives.

---

## Decision record

Quality bar, in order:

1. `nvidia/GLM-5.3-Flash-NVFP4` — ModelOpt W4A4, recipe `nvfp4_experts_dense_mlp-kv_fp8_cast`, producer `modelopt 0.47.0.dev393+ga4bc45b30.d20260828`. **This plan.**
2. `RedHatAI/GLM-5.3-Flash-NVFP4` — compressed-tensors. **Not this plan.**
3. EXL3 / TR3 — **out of scope.**

The model card's sentence that only shared experts and dense MLPs are quantized does not match the index. Routed experts and the three dense MLPs carry scales. Shared experts, attention, the router, embeddings, `lm_head`, and layer 45 do not. Trust a manifest-pinned index, not the card and not a floating `main`.

One copy of the routed experts is about 171 GB. Two Sparks have 256 GB of unified memory together. Two resident copies do not fit, and CPU memory on those nodes is the same pool. The existing vLLM serve stays up. Comparison is record, then a later TensorFold run, inside an approved window that says how the previous serve is restored.

### Reviews already accepted

Checked against the pin. Still in force:

- Expert downs are distinct intermediates. `matmul_group` cannot run them.
- Graphs need device-side weight selection. Recapture-per-token and copying selected weights into the capture are rejected.
- Prefill packs assignments and scatters back. It does not run every row through every expert that appears in the chunk.
- `--no-drafts` does not unload MTP. `MTP_DEFAULT` is `"1"`. `serial_only` matters only for `TF_GLM_MTP=auto`.
- Bitwise equality is only for the same backend and the same reduction order. `qmm.split_k(n, k)` changes when a split changes `n` or `k`.
- `matmul` copies a rejected destination back. `matmul_group` does not. `_out` drops a noncontiguous or wrong-dtype buffer.
- Column-split rule: original packed width `P` satisfies `P % 64 == 0`, so each rank's logical K (`P`) is a multiple of 64. `P = 96` passes `% 32` and then fails `Fp4Linear`.
- ~80 tok/s was a MoE-only partial bound. ~31 tok/s is a larger partial bound. Neither is a throughput target.
- A 32k timing run needs an explicit `--context`. The default window stays dense.
- No new MMA, no CUTLASS/FlashInfer dependency, no "Four Over Six" rescale (arXiv:2512.02010).

### What the third review adds

Checked against `checkpoint.cpp` `lane` (one matrix, one alpha), `nvfp4q.cuh` `block_scale` (SATFINITE e4m3, multiply by zero when the scale byte is zero), `forward.Buffers` (`slots = top_k + 1`, prefill `ey` is bf16), and `geometry.py` (draft-head term `9/16` of a bf16 head, file is `src/tensorfold/cuda/geometry.py`).

Accepted:

- One owner for expert storage. Pointer table first. No second stacked copy.
- A small CUDA/C++ dispatch wrapper is allowed and required. It is not a new MMA.
- Prepared-row layout is an interface: logical M/K, padded M, code layout, scale layout, lifetime.
- Prefill capacity is the worst-case assignment count plus tile padding, not the average of ~57 rows per expert.
- Eight routed slots and the shared expert's ninth slot are different. The shared expert is not an entry in the NVFP4 table. Its combine weight is 1.
- Decode and prefill each have their own eager oracle. A lane/prompt difference is not a scheduler bug.
- Today's ModelOpt `main` is not the producer (`0.47.0.dev393`) and is not the vLLM activation kernel. Record both policies. Do not re-quantize stored weights to match a newer encoder.
- SwiGLU rounds `g * sigmoid(g)` to bf16, then multiplies by clamped `u`, then rounds the product. It does not merely round the sigmoid.
- Two graph tests: a primitive with supplied ids, and a full forward whose router produces the ids. Rows 1 and 4 are not the whole capture set (`GRAPH_ROWS` is 1..6).
- Unset `TF_GLM_MTP` and explicit `1` stay distinguishable. Unsupported drafting is rejected before NCCL.
- Top-k logprobs are not a full-vocabulary oracle. Missing evidence is `INSUFFICIENT_EVIDENCE`, not `MATCH`.
- `m > 2ε` is a sufficient condition that the top token is stable. The converse is not a near-tie diagnosis.
- Reference layer traces are captured before the reference is unloaded.
- Admission accounts for the real NVFP4 allocations and drops the draft-head term this path does not allocate.
- Launches use the manifest's snapshot, not a floating repository id. A `Range` request that returns a whole shard is aborted.
- Qualification is a set of scoped statuses, not one `MATCH` bit.

Narrowed:

- Compute Sanitizer runs on the new dispatch and the packing kernels, on tiny fixtures. A skipped sanitizer is recorded, not treated as a pass. It is not required on every CPU commit.
- Full prefill graph capture is not implied by decode graphs. The manifest says which prefill calls are captured.
- CPU streams do not wait on a live reference. `BLOCKED_REFERENCE` stops parity claims, not loader work.
- GPU kernels and real NCCL do not run on the machine that is serving the live model unless that window was approved.

---

## What the checkpoint stores

Measured 2026-10-02 from the public index, safetensors headers, and the leading 8 KiB of `input_scale` values in shard 1. No shard body was pulled. These are plan facts until Task 1 checks the pinned revision. A sample of shard 1 does not prove every shard.

Export: https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4

- 147,661 tensors, 33 shards, `total_size` 204,419,110,596 bytes (190.4 GiB).
- `quant_method` `modelopt`, `quant_algo` `NVFP4`. The **global** activation factor is static. Group size 16. Per-16 e4m3 block scales of an activation are computed at runtime. They are not stored.
- `kv_cache_scheme`: static FP8, no scale tensor in the index.
- 45 layers: 34 `linear_attention`, 11 `deepseek_sparse_attention`. 3 dense MLPs, then 42 MoE layers. Hidden 4096, dense intermediate 12288, expert intermediate 2048, 288 routed experts, 1 shared, top-8, `routed_scaling_factor` 2.5, `norm_topk_prob` true, scoring `sigmoid`, `moe_router_dtype` `float32`, `swiglu_limit` 10. Linear attention 64 heads of dim 128. `qk_rope_head_dim` 0, `kv_lora_rank` 512, `index_topk` 2048, vocab 154880, context 1,048,576, one next-n layer.

Codes are `.weight` dtype `U8`. There is no `weight_packed` and no `weight_global_scale`. The global weight scale is scalar `weight_scale_2`. The global activation factor is scalar `input_scale`. Gate and up may share `input_scale` and must keep distinct `weight_scale_2` values. An early shard sample looked like one gate scale per layer. Language-model layer 3 does not: expert 0 is `5.8128720411332324e-05`, expert 1 is `4.650297705666162e-05`, expert 2 is `4.3596541217993945e-05`. Each expert keeps its own pair in its alpha slot. Two shards of the same expert must still agree. That disagreement stops the load.

| Tensor | Stored shape | Logical GEMM |
| --- | --- | --- |
| Expert gate/up `.weight` | `U8 [2048, 2048]` | `[2048, 4096]` |
| Expert gate `.weight_scale` | `F8_E4M3 [2048, 256]` | one e4m3 per 16 K |
| Expert down `.weight` | `U8 [4096, 1024]` | `[4096, 2048]` |
| Expert down `.weight_scale` | `F8_E4M3 [4096, 128]` | |
| Dense layer-0 gate `.weight` | `U8 [12288, 2048]` | `[12288, 4096]` |
| Dense layer-0 down `.weight` | `U8 [4096, 6144]` | `[4096, 12288]` |
| `weight_scale_2`, `input_scale` | `F32 []` | replicate |
| Shared expert gate | `BF16 [2048, 4096]` | not NVFP4 |
| Router | `BF16 [288, 4096]` in the file | scores are fp32 |
| Layer 45 expert gate | `BF16 [2048, 4096]` | 889 tensors, none scaled |

After TP=2, expert gate/up N is 1024 and expert down logical K is 1024. Dense down logical K is 6144. All of those are multiples of 64. `cfg.group_size` still defaults to 64 when the key is absent. Do not read it as 16.

### Dataflow

`R` tokens, one rank. Routed width after the row split is 1024.

| Object | Shape |
| --- | --- |
| Residual | `[R, 4096]` |
| Routed ids and weights | `[R, 8]` |
| Shared slot | index 8, or a separate buffer. Not an NVFP4 expert id |
| Gate/up | `[R, 8, 1024]` |
| Clamped SwiGLU | `[R, 8, 1024]`, one vector per assignment |
| Routed downs | `[R, 8, 4096]` fp32 |
| Combine | slot order, shared contribution last at weight 1, into fp32 `b.part` |

SwiGLU, matching `glue.swiglu`:

```text
g1 = min(g, 10)
u1 = clamp(u, -10, 10)
s  = round_to_bf16(g1 * sigmoid(g1))
h  = round_to_bf16(float32(s) * u1)
```

Dense, shared, and routed paths each get a saturation test. `mlp_prompt` stays off: it runs epilogue 2, which does not clamp and keeps the product in fp32. Epilogue 1 rounds `u` to bf16 first, which is also not this sequence.

Prefill of 2048 rows has `2048 * 8 = 16384` assignments. Even routing is about 57 rows per expert. That is not the allocation bound. One expert can receive all 2048 rows. With tile padding `B`, a loose bound is `16384 + 288 * (B - 1)` (about 34,528 rows at B=64, about 52,960 at B=128). Admission uses the bound of the tile policy actually selected. A fixed graph may contain empty work descriptors. An empty expert does no matrix work and reads no expert rows. "Zero host launches" is not the requirement.

Eight fp32 slots at R=2048 are 256 MiB. Nine are 288 MiB. The existing nine-slot bf16 `ey` is another 144 MiB if it stays allocated. Pick one representation in Task 11A and do not add 256 MiB on top of the old buffer without saying why both exist.

---

## Numeric policy

Write this to `docs/plans/notes/nvidia-glm-nvfp4-numeric-policy.json` in Task 0A. Implementation checks the file; it does not invent a tolerance after seeing the candidate.

| Stage | Policy |
| --- | --- |
| Stored codes and scales | Exact after packing and splitting. Padding is separate. |
| Global activation factor | Stored `input_scale`. Not recomputed from the activation. |
| Runtime block scales | TensorFold `block_scale`: e4m3 via SATFINITE, multiply by zero when the scale byte is 0. ModelOpt `main` clamps tiny block scales toward `2**-9` before the cast. Those policies are named, not assumed identical. A zero block that is byte-different but numerically zero is not by itself a broken load. A nonzero-value difference is investigated before parity is claimed. |
| Gate/up | FP4×FP4, bf16 production output. |
| Down | FP4×FP4, fp32 partial. |
| Combine | Selected-slot order, shared last at weight 1. |
| TP sum | fp32 partials, rank 0 first, as `gather` / `hc_post` do now. |
| Head | Record compute dtype, vocab order, and padding. Widening bf16 to fp32 for storage is labeled as widening. |
| Cache | bf16 latent unless a later task qualifies a known fp8 writer. |

Two reference levels, kept apart:

1. A small logical E2M1/E4M3 encoder in the test, with its rounding and zero policy written down. Not a second call to `quant4`. Not a copy of today's ModelOpt `main` treated as the producer.
2. The eager TensorFold kernel on the same backend as the candidate (lane for decode, prompt GEMM for prefill).

A high-precision dot product of the **decoded quantized operands** is a primitive reference. It is not a substitute for the runtime kernel, and an unquantized bf16 model is not the W4A4 oracle.

| Comparison | Rule |
| --- | --- |
| Checkpoint to packed weights | Exact codes, scales, and which scalar is which |
| Eager to device dispatch, same backend | Bitwise |
| Eager to graph, same state | Bitwise |
| TP split to unsplit | Named envelope. `split_k` recorded on both shapes. Not bitwise |
| Decode to prefill | Envelope, unless the test shows the same reduction |
| TensorFold to vLLM | Same inputs, observation type named, envelopes frozen before the candidate is scored |

Let `ε` be the max absolute error over the **full** observed vocabulary, and `m` the reference top-1 minus top-2 margin. If `m > 2ε`, the candidate must keep that top token. If `m <= 2ε`, the bound is inconclusive. It is not evidence of a benign near-tie. A near-tie label requires the error itself to sit inside an envelope that was fixed before this run, and no earlier defect. Any emitted-token difference still fails token parity. A top-k logprob vector is not `ε`.

Router checklist, because a miss here is not an NVFP4 bug: bf16 matmul into fp32 scores, sigmoid, bias added for the choice only, lower-id tie break, normalize unbiased scores when `norm_topk_prob` is set, then `routed_scaling_factor`. Both ranks must pick the same ids. Do not broadcast one rank's ids to hide a mismatch. mHC constants, KDA state, indexer parameters, and head dtype are on the same checklist. Record code defaults, not only `config.json`.

---

## Capacity and traffic

Calculations, not measurements. DGX Spark: 128 GB unified, 273 GB/s ([hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)). The 273 GB/s figure is a spec, not a rate every kernel sustains.

| Quantity | Result |
| --- | --- |
| One routed expert, codes plus block scales | 14,155,776 bytes |
| Routed experts resident per rank | 79.734375 GiB |
| One MoE layer of those, per rank | 1.8984375 GiB |
| Routed traffic, top-8, 42 layers, per token per rank | 2.378 GB |
| Routed plus shared expert traffic | 3.435 GB/token/rank |
| Partial token model in the table below | 8.859 GB/token/rank |
| Draft-head term this path must not keep | about 170 MiB/rank (`vocab/2 * 4096 * 9/16`) |
| 256 full-vocab fp32 vectors on disk | 151.25 MiB |

| Component | GB/token/rank |
| --- | ---: |
| Top-8 routed experts, 42 layers | 2.378 |
| BF16 shared experts, 42 layers | 1.057 |
| Three dense NVFP4 MLPs | 0.127 |
| BF16 head (vocab/2) | 0.634 |
| BF16 routers | 0.099 |
| BF16 KDA Q, K, V, O, 34 layers | 4.563 |
| Partial sum | 8.859 |

`273 / 8.859 ≈ 31` tokens/s only if that list were the whole token and every byte moved once at the spec rate. The 11 sparse-attention projections, the smaller KDA projections, mHC, the latent, and about 90 collectives of 16 KiB are not in the sum. Report a modeled byte-rate only with its numerator. Do not call `tok/s * 2.378e9 / 273e9` an efficiency unless the timer covered only those expert GEMMs.

Layer 45, if loaded, is about 6.75 GiB/rank of BF16 expert weights. This plan does not load it.

Eleven sparse layers, 32,768 tokens, latent width 512, replicated per rank: about 352 MiB bf16, about 176 MiB if the latent were fp8. Expert-slot scratch at R=2048 can exceed that saving. `mla_geometry` does not divide the latent by world.

---

## Evidence

Observation types, and nothing else:

`TOKEN_IDS_ONLY`, `TOPK_LOGPROBS`, `FULL_VOCAB_LOGPROBS`, `FULL_VOCAB_RAW_LOGITS`, `INTERNAL_LAYER_TRACE`.

Missing fields are `INSUFFICIENT_EVIDENCE`. The comparator does not retokenize text, does not fill missing probabilities with zeros, and does not treat top-k logprobs as a full-vocabulary max error. Full logprobs are not raw logits. Widening bf16 head outputs into fp32 is labeled.

Statuses, separate fields: `CONFIG_PASS`, `LOAD_PASS`, `PRIMITIVE_PASS`, `ROUTED_PASS`, `GRAPH_PASS`, `TP_PASS`, `DENSE_FIDELITY_PASS`, `LONG_CONTEXT_FIDELITY_PASS`, and failures `DIVERGE`, `INSUFFICIENT_EVIDENCE`, `BLOCKED_REFERENCE`, `UNSUPPORTED_MODE`. A token-parity pass with no full-vocabulary observations does not publish a recipe claim. Diagnostic timings before fidelity are allowed and labeled unqualified.

Package layout (Task 0A creates the readers; Task 12 fills a real package only in an approved window):

```text
reference/<manifest_id>/
  manifest.json
  numeric_policy.json
  capture_capabilities.json
  observations.jsonl
  tensors/<case>.<stage>.<position>.<rank>.safetensors
  checksums.sha256
```

No pickle. Synthetic or public prompts only. Before a reference process exits, the package that was actually authorized includes the boundaries the comparator will need later: router ids and weights, selected gate/up, post-SwiGLU, down partials or the combine, and the head vector type that the server can really return. A missing probe is written down. It is not reconstructed after the process is gone.

First-difference order when a compare fails: checkpoint bytes, then activation codes, then same-operand GEMM, then device ids versus eager, then prefill versus decode, then graph versus eager, then rank disagreement, then the reduction, then anything upstream of the quantized MLP (KDA, norms, mHC, cache), then repeated-request state, then long context, then the head. Do not start at the KV writer. Do not call a short dense match a sparse-attention qualification.

---

## Work streams

One orchestrator. The other roles are sub-agents with a file list. A coder does not review their own change. A reviewer reads the diff against this document and does not quietly add a feature. A tester owns the test file, runs the command, and reports a status from the list above. GPU work does not use the GPUs of the live serve unless the orchestrator has an approval that names the restore steps.

The orchestrator may start a stream only when its "needs" line is met. Streams in the same window edit disjoint files. `weights.py` and `forward.py` are sequential: the policy stream lands the refusal, the load stream replaces the loader branch, the numerics stream wires eager forward, the kernel stream wires the device path. Two coders do not edit those files at once.

| Stream | Role | Owns | Window | Needs |
| --- | --- | --- | --- | --- |
| O | Orchestrator | `docs/plans/notes/*manifest*`, status record, merge order | 0 onward | nothing |
| E | Evidence coder, then tester | `tools/glm_nvfp4_record.py`, `tools/glm_nvfp4_compare.py`, schema tests | 0 | nothing |
| P | Policy coder, reviewer, tester | `glm5_next/__init__.py`, new `cuda/nvfp4_policy.py`, `tests/cuda/test_glm5_nvfp4_config.py` | 0–1 | Task 0 identities for the manifest fields; Task 2 can start on today's tree |
| C | Census (orchestrator or one fetcher) | `docs/plans/notes/nvidia-glm-nvfp4-index.md` | 1 | Task 0 checkpoint revision |
| R | Recon, read-only | feasibility note only | 1 | a reference that is already up, or an honest `BLOCKED_REFERENCE` |
| S | Split coder, tester | `cuda/split.py`, `tests/cuda/test_glm5_nvfp4_split.py` | 2 | Task 1 shapes |
| Q | Quant coder, tester | `tests` for the logical encoder, `cuda/nvfp4` prepared-row helper, one-projection loader | 2–3 | Task 4 for real shapes; 5A can start with no loader |
| M | Memory designer | `src/tensorfold/cuda/geometry.py` inventory, no kernel edits | 2, before Task 6 | the pointer-table decision in this plan |
| L | Load coder, tester | `cuda/nvfp4_load.py`, the `modelopt` branch of `cuda/weights.py` | 3 | Tasks 3, 5C, 11A |
| N | Numerics tester, with a coder for `forward.py` eager path | `cuda/nvfp4_moe.py` oracle, eager tests | 3–4 | synthetic `Fp4Linear` from Q; full layer from L |
| K | Kernel coder, reviewer, sanitizer tester | new dispatch `.cu`/binding next to `checkpoint.cpp`, not a new MMA file that reimplements `mma_fp4` | 4 | Task 5B layout, Task 8 projection oracle |
| D | Distributed tester | two-process tests on the existing comm wrapper | 4 | approval if those processes need the live GPUs; otherwise a CPU `comm=` seam |
| G | Graph tester | `cuda/graphs.py` only if capture assumptions break, plus graph tests | 5 | Task 10B |
| Pub | Orchestrator | recipe text | last | structured statuses, not a Markdown `MATCH` |

Windows:

0. O writes the manifest. E builds the comparator on synthetic records. P writes the refusal tests and the CPU policy function. No GPU, no shards.
1. C confirms the census. R inspects a reference only if that does not restart it. P admits the recipe. E stays on synthetic failures.
2. S splits. Q writes the logical quantizer (5A) in parallel. M writes the allocation inventory (11A) before anyone stacks tensors.
3. Q finishes prepared rows and one projection. L loads experts into the pointer table. N writes eager oracles on synthetic linears.
4. K lands one device-indexed projection and runs sanitizer on a tiny fixture. N extends oracles to unequal experts. D runs the two-process smoke test when GPUs are allowed.
5. K finishes routed decode and packed prefill. G runs primitive and full-forward graph tests.
6. Approved window only: record the reference, unload only as the approval says, run the candidate, restore the previous serve.
7. Long context, then scoped timings. Publication reads the status record.

A reviewer blocks a window when any of these is true: a second copy of expert storage, `matmul_group` on expert downs, `experts.cu` on this path, MTP loaded, a tolerance raised to fit the candidate, a `MATCH` from top-k logprobs, a graph test that only pokes ids after the router has already overwritten them, or a launch of floating `main`.

---

## Tasks

Each task ends in a commit of code and tests, or an evidence note. A blocked GPU gate can commit a report. It cannot turn on an unqualified serve.

### Task 0 — Manifest

Orchestrator. No shard pull.

`docs/plans/notes/nvidia-glm-nvfp4-manifest.md` and a JSON twin. Fields: base commit `56e2e3e…`, `HEAD`, dirty flag, patch hash if dirty, checkpoint commit from the Hub, sha256 of `config.json` and of the index, tokenizer hash, world size 2, cache policy unknown, reference status `NOT_INSPECTED`. Unknowns are null with a reason.

The parser rejects a missing identity, a hash mismatch, world size other than 2, and a numeric policy that contradicts the table above.

```bash
git commit -m "docs: pin the GLM NVFP4 tree and checkpoint revision"
```

### Task 0A — Comparator

Stream E. No weights, no CUDA.

Schema, observation enum, and comparator. Synthetic tests: full-vocab logits, full logprobs, top-k only, missing token ids, nonfinite values, vocab mismatch, truncated file, a large corrupt logit vector that satisfies `m <= 2ε` and must not be labeled a near-tie. Top-k evidence fails a requested full-distribution gate. Text is never retokenized into ids.

```bash
git commit -m "test: offline comparator for GLM NVFP4 reference records"
```

### Task 0B — Reference feasibility

Stream R. Read-only. If the running server is a different checkpoint, it is not the oracle: write `BLOCKED_REFERENCE` and the exact gap. If it is this export, record image digest, package versions, GPU capability, clamp, cache, the MoE backend line for prefill and for decode, and whether activations stayed FP4. Absence of a log line is not proof of a backend. Do not restart it.

CPU streams continue either way. Parity does not.

### Task 1 — Census

Stream C. Re-fetch the index and `config.json` at the manifest revision. If a `Range` response is the whole shard, abort. Confirm 147661 tensors, 33 shards, `total_size` 204419110596, `weight_packed` 0, layer 45 present and unscaled, and the representative shapes in the table (dense, routed, shared, router, head). Header and the small `input_scale` sample are not a full checksum.

A count change stops loader work until this plan is updated. Do not delete caches to make room.

```bash
git commit -m "docs: census nvidia GLM-5.3-Flash NVFP4 tensors"
```

### Task 2 — Lock refusals, CPU seam

Stream P. Complete `Config.read` fixture using `quantization_config`. Today's `require_readable`, `check()`, and `load` still refuse `modelopt`. Add `nvfp4_policy.py` with a pure function of the raw environment and the flags, so tests never construct `GlmEngine`. A missing unrelated config field must not make a refusal look like a pass.

```bash
git commit -m "test: lock GLM CUDA refusal of ModelOpt NVFP4"
```

### Task 3 — Admit this recipe only

Stream P. `QUANT_METHODS["cuda"]` gains `"modelopt"` only. Update `tests/cuda/test_glm_split_and_policy.py`. Accept static NVFP4, group 16, float weights and activations. Reject compressed-tensors, W4A16, integer codes, a wrong group, dynamic activations, and a non-finite global factor. `load` still raises `NVFP4 tensors are not wired`. Do not fall through into a bf16-activation expert path.

```bash
git commit -m "feat: admit the NVIDIA NVFP4 recipe in the GLM family gate"
```

### Task 4 — Splits

Stream S. Scalars return `"rep"` before the row/column scan. `F8_E4M3` in both dtype maps. Original packed width `P % 64 == 0` for a TP=2 column split. `P = 96` raises. Do not pad.

Tests: the real gate/up and down shapes, distinct bytes in each K group, non-unit scalars staying `[]`, `split_bytes` and `split_device` agreeing, and a rank folder whose metadata records source revision, format version, world, rank, shapes, dtype, axis, and scalar policy. Wrong rank, wrong world, stale revision, and truncated files fail before GPU work. CPU, device, and folder paths reconstruct the same codes and scales. Run `tests/test_cuda_capacity.py` and `tests/test_cuda_geometry.py`.

```bash
git commit -m "feat: split GLM NVFP4 tensors on groups the kernel can use"
```

### Task 5 — Quantizer, prepared rows, one projection

Stream Q.

5A. Logical encoder in the test. Cases: zero blocks, tiny maxima, subnormal-scale edges, E2M1 ties, negatives that round to zero, saturation, non-unit globals, nonfinite inputs. Failures say whether codes, scales, or reconstructed values moved. Do not treat ModelOpt `main` as the producer.

5B. `PreparedRows`: logical rows, logical K, padded rows, code layout id, scale layout id, storage, activation factor, owner. `quantize_into` an explicit buffer. Round-trip at a nonzero offset, a padding boundary, and two rows with different block scales. Reject an unknown layout. Do not gather scale axis 0 as if it were the row axis.

5C. One dense projection via `Fp4Linear.from_checkpoint`, `act=input_scale`, `scale=weight_scale_2` not inverted. Gate and up with equal `act` and unequal `weight_scale_2` stay distinct. Missing global scale raises. `model.visual` skipped. Expert names raise `routed experts are not wired` until Task 6.

5D. One real local shape on the lane kernel and one on the prompt kernel. Record dtype, `split_k`, and the envelope. No whole-model load.

```bash
git commit -m "feat: map one GLM dense MLP projection to Fp4Linear"
```

### Task 11A — Allocation inventory, before Task 6

Stream M. Commit the inventory before the loader allocates expert tables. Pointer table, not a second stack. List final weights, load scratch, prepared rows, route metadata, gate/up, routed fp32 slots, shared fp32 output, split-K partials, graphs, collectives, and headroom. Remove the draft-head `9/16` term for this quant. Choose eight-plus-separate-shared or nine fp32 slots, and say whether the old bf16 `ey` remains. File is `src/tensorfold/cuda/geometry.py`.

```bash
git commit -m "docs: inventory GLM NVFP4 allocations before the loader owns them"
```

### Task 6 — Load into that storage

Stream L. Prefetch `weight`, `weight_scale`, `weight_scale_2`, `input_scale` for routed projections, and `.weight` only for the shared expert. Attention, norms, router, embed, and head use the BF16 arms. No `draft_head`. No layer 45. Eager views and the address table share backing storage; a test checks identity by data pointer, not by summing `nbytes`. E=4 is the numeric fixture. An E=288 test checks addresses without allocating production weights. A failed load frees only what it allocated.

`Weights.nbytes` counts an `Fp4Linear` and does not double-count a shared buffer.

```bash
git commit -m "feat: load GLM routed experts into a single-owner table"
```

### Task 7 — Startup before NCCL

Stream P, in `nvfp4_policy.py`. Raw environment, so unset is not the same as explicit `1`.

| Setting | Result |
| --- | --- |
| Unset, `0`, or `auto`, with `--no-drafts` | No layer 45, no draft head, no MTP graph |
| Explicit `1` | Raise `unqualified` before `set_device` and NCCL |
| Drafter, drafts on, or any parallel mode | Raise before NCCL |
| Bad value | Raise |

MLX and EXL3 keep `MTP_DEFAULT == "1"`. Vision stays the existing `serve_options` error. Negative tests assert the CUDA and NCCL constructors were not called. When two ranks do run, they exchange a digest of revision, world, rank, context, kernel policy, and cache mode, and a missing peer fails in finite time. `engine.py` does not contain `drop --tp 2` or `run on one GPU`. Qwen and Flash Next still refuse NVFP4 `--tp 2`.

```bash
git commit -m "feat: GLM NVFP4 refuses MTP before NCCL and keeps two ranks"
```

### Task 8 — Eager oracles

Stream N.

Dense: common-input quant for gate and up only when their activation factors match, then `glue.swiglu`, then one down into the caller's fp32 buffer. A gate value above 10 differs from `mlp_prompt`.

Routed: unequal expert weights and unequal `weight_scale_2`. Each down reads its own intermediate. Shared expert stays on the BF16 MLP and still contributes if every routed weight is zero. Highest expert id and the shared slot are separate cases.

Two oracles: lane/decode and prompt/prefill. Do not compare them bitwise. Same-backend bitwise is the gate.

Destination tests: noncontiguous multi-row view, wrong dtype, canary past the write, overlap. A refused buffer raises or is copied back. It is not silently replaced.

```bash
git commit -m "feat: run GLM NVFP4 eager projections with per-expert downs"
```

### Task 9 — TP arithmetic, then real ranks

9A, stream N. Row-split gate/up by concatenation, column-split down by an ordered fp32 sum. Record `split_k` on the full shape and the rank shape. Envelope, not bitwise.

9B and 9C, stream D, only when the GPUs are free or the test uses the `comm=` seam. Small tensors, eager and captured, rank order, repeated calls, both ranks in the capture. A missing peer and a mismatched digest fail and clean up. Both ranks' expert ids are recorded. Do not broadcast one side's ids.

```bash
git commit -m "test: GLM NVFP4 two-rank partials and a two-process smoke test"
```

### Task 10 — Device dispatch, then graphs

Stream K, then G. Four commits.

10A. One projection. New binding beside `checkpoint.cpp`. Reuse `mma_fp4` and its split-K grouping. 64-bit address offsets. Same-backend bitwise against the eager projection, including a second id on replay. Compute Sanitizer memcheck and initcheck on a tiny fixture. A skipped tool is a limitation, not a pass.

10B. Full routed decode: one residual quant, eight experts, clamped SwiGLU, eight distinct intermediate quants, downs, shared branch, ordered combine. Asymmetric experts. Saturating activations. Shared-only contribution.

10C. Prefill: counts, exclusive prefix sums, stable pack, tile descriptors sized for the worst case, scatter to `[token, slot]`, then the deterministic combine. Do not atomic-add fp32 outputs in completion order. Tests: all tokens on the same eight experts, balanced, empty, last id, counts around the tile, every row a different route, large then small reuse. Padding writes nothing a user can see. Unselected experts do no logical work.

10D. Primitive graph: supplied ids change, table addresses do not. Full-forward graph: change the input so the real router changes the route, and check it is not overwritten back. Every row count the serve can replay, both parities, and the sparse bucket transition if that path is reachable. If serial NVFP4 supports fewer rows, refuse the others at startup. Sequences: A then B then A, large then small then large, graph then eager then graph. Capture leaves a clean recurrent state. Say in the manifest whether prefill is eager. Do not call eager prefill graph-qualified.

```bash
git commit -m "feat: index GLM NVFP4 experts from a device table"
```

### Task 11B — Measure the inventory

Stream M, after each GPU milestone and again before a full load. Compare the dry-run geometry to unique allocations, not to the sum of views. Log peak load, post-load, post-capture, and the largest prefill. A model that fits after load and cannot survive the admitted prefill fails this gate. Do not infer the unified-memory budget from one API.

```bash
git commit -m "feat: reconcile GLM NVFP4 admission with measured peaks"
```

### Task 12 — Record, then compare

Only with approval, and only after `PRIMITIVE_PASS`, `ROUTED_PASS`, `GRAPH_PASS`, `TP_PASS`, and a dry-run admission for the exact context. The orchestrator does not start this because the calendar is clear.

12A. Reference capture of this checkpoint revision. Repeatability of the reference itself first. Golden package from Evidence. Timing from an uninstrumented process, traces from a labeled instrumented one.

12B. Hashes and schema checks before that process exits.

12C. Candidate loads the manifest directory, not `nvidia/GLM-5.3-Flash-NVFP4` as a floating name. Smoke the France prompt as `TOKEN_IDS_ONLY`. Then the fixed teacher-forced corpus, prefill and incremental, with position semantics written on each record (`prefix length t predicts token t` or otherwise).

12D. Earliest recorded boundary. Router ids on both ranks. If a probe is missing, report the interval.

12E. Write the status JSON. Restore the previous serve whether or not the candidate passed.

`DENSE_FIDELITY_PASS` needs the numeric gates, input identity, the observation coverage the manifest required, and the path tests. Token parity is its own field.

```bash
git commit -m "test: compare GLM NVIDIA NVFP4 to a saved reference"
```

Commands, inside that window:

```bash
TF_GLM_MTP=0 tensorfold serve <manifest-snapshot> --tp 2 --rank 1 --master RANK0 --no-drafts
TF_GLM_MTP=0 tensorfold serve <manifest-snapshot> --tp 2 --rank 0 --master RANK0 --no-drafts --host 127.0.0.1 --port 8080
```

No `--context` here. This is the dense window. vLLM, when recorded, is `--tensor-parallel-size 2`, greedy, no speculative decoding. Not the card's TP 4.

### Task 13 — Long context

After `DENSE_FIDELITY_PASS`. Both ranks get `--context` at least prompt plus generated tokens, and the log must show that effective context. Before any 32k timing: one step around `dense_limit`, the prefill chunk boundary, and a sparse pool boundary. Cold, reset, and a continuation after an unrelated request. KDA conv and recurrent state, not only the latent. Prefix reuse only if the engine already has it. FP8 latent only after the reference writer is identified. A mixed-cache run can localize a bug and cannot pass same-cache parity.

```bash
git commit -m "test: qualify GLM NVFP4 past the dense attention window"
```

### Task 14 — Timings, one change at a time

After the fidelity status that the sentence will claim. Unqualified diagnostic timings from earlier windows stay out of the recipe.

Report streaming time to first token, generated-token count, decode-only tok/s, median and tail inter-token time, warmup excluded, cold and warm prefix separately if both exist, context, chunk, peak memory, and power if the machine reports it. Load and capture are not part of steady state. Do not add overlapping profile ranges and call the sum the token.

Profile routed experts, shared experts, KDA Q/K/V/O, other attention, the head, collectives, and leftover. Label numerators as in the traffic section.

An optimization record: hypothesis, exact change, invariant, expected effect, tests required, memory effect, measurement, outcome, rollback. One mechanism per record. A fusion or a `split_k` change is a new numeric policy and reopens the oracle.

The recipe may say dense-window single-request parity, or long-context parity, only when that status is set. It does not say MTP, vision, expert parallel, concurrency, or that this replaces the existing server.

```bash
git commit -m "docs: record GLM NVIDIA NVFP4 qualification and scoped speed"
```

---

## Non-goals

- No new MMA. The dispatch wrapper is in scope.
- No CUTLASS, FlashInfer, or TensorRT-LLM dependency.
- No `experts.cu` on this checkpoint.
- No EXL3 work, no RedHat loader, no Apple NVFP4.
- No vision, no DFlash2, no layer-45 load. `incoai/GLM-5.3-Flash-DFlash2` is CC BY-NC-ND 4.0.
- No expert parallelism, no `--parallel`, no second resident copy.
- No "Four Over Six" rescale.
- No published speed before the matching fidelity status.

## Stop conditions

- The pinned census moves, or layer-wide activation scales disagree across shards.
- A split cuts a group of 16, leaves logical K not a multiple of 64, or splits a scalar.
- Expert storage is allocated twice.
- The device path does not match its same-backend eager oracle bitwise, including a second route and a skewed chunk.
- The only graph strategy left is recapture-per-token or copying selected weights.
- A full-vocabulary or layer-trace claim is made from top-k logprobs.
- A tolerance is widened so the candidate passes.
- Continuing requires stopping the live serve before the comparator and the restore plan exist.
- The work needs a multiply other than `lane4.cu` / `gemm_ck.cu` / `gemm_ws.cu`.

## After this plan

RedHat compressed-tensors, after `DENSE_FIDELITY_PASS`, reusing the Qwen reciprocal global scale. MTP on the BF16 layer-45 experts. A multi-request serve. A clamped SwiGLU fusion only after it matches `glue.swiglu` bitwise on the same rows.

## Checklist

CPU, done on this branch unless noted:

- [x] Manifest names the plan commit, the checkpoint revision `da920bb`, and hashes. Launches still must use that snapshot once it exists. `implementation_head` in the JSON is the plan commit `d3b7b07`, not every later commit.
- [x] Comparator tests pass on CPU, including the false near-tie and the top-k refusal. Reference status is `BLOCKED_REFERENCE`.
- [x] Census matches the table.
- [x] `tests/test_glm5_nvfp4_config.py`, `tests/test_glm5_nvfp4_split.py`, `tests/test_glm5_nvfp4_codec.py`, and `tests/test_glm5_nvfp4_inventory.py` pass. The `load()` refusal and `split_device` passed on one Spark. `tests/test_glm5_nvfp4_loader.py` does not exist yet.
- [x] MLX/EXL3 `MTP_DEFAULT` is still `"1"` (`tests/test_glm_mtp_setting.py`).
- [ ] Qwen and Flash Next still refuse NVFP4 `--tp 2`. `tests/cuda/test_glm_split_and_policy.py` passed on one Spark (10). That file does not cover Qwen or Flash Next.
- [x] Explicit `TF_GLM_MTP=1` on this quant raises before the engine import. Modelopt forces `mtp_on` false inside `GlmEngine`. The snapshot two-rank digest is not built. The small-tensor smoke recorded a revision mismatch.
- [x] Prepared-row test fails if the scale layout is treated as row-major. Scale axis 0 is a K group.
- [x] Inventory says one pointer table, nine fp32 slots, no draft head, no second copy. Rank 0 later allocated 95,144,077,812 bytes and was dropped.

Spark, not done:

- [x] `split_device` matches `split_bytes` on a GPU.
- [x] One dense projection is an `Fp4Linear`. Lane and prompt both returned finite bf16. `split_k` for `N=6144`, `K=4096` is 2. Same-backend eager oracles passed. A numeric envelope is not recorded.
- [x] Synthetic E=4 eager owners match the address table by `data_ptr`. Device dispatch landed. That is not a production-snapshot identity check.
- [x] Expert downs are not `matmul_group`. Unequal weights and unequal `weight_scale_2` fail the test if a down is grouped with another expert. The shared slot is separate from the highest expert id.
- [ ] Primitive graph follows a supplied-id change. Full-forward graph does not. The real router is not in that capture.
- [ ] Prefill worst-case capacity is in the live admission path. Empty experts do no logical work. 10C tests the pack. It is not graph-qualified.
- [x] Small-tensor two-rank digest, missing peer, and captured ordered sum passed. A snapshot two-rank load has not.
- [x] Sanitizer zero errors on the 10A canary only. Other shapes are an explicit limitation, not a pass.
- [ ] The pinned snapshot is local. The previous serve was not restored. No second resident copy. No recipe claim.
- [ ] Recipe text matches a status record. `DENSE_FIDELITY_PASS` is not `LONG_CONTEXT_FIDELITY_PASS`.

## References

Format and export:

- NVIDIA NVFP4 introduction: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Model card (recipe name, TP4 example, GB200 test note; the shared-expert sentence is not followed): https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4
- ModelOpt `NVFP4QTensor` (global factor `amax / (6 * 448)`, block-scale clamp on current `main`): https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py
- ModelOpt PTQ recipes: https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md
- OCP MX v1.0 (E8M0, group 32). This export is not that format: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- TensorRT-LLM quantization table: https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html

Engines:

- vLLM NVFP4 oracle 0.22.1: https://docs.vllm.ai/en/v0.22.1/api/vllm/model_executor/layers/fused_moe/oracle/nvfp4/
- vLLM oracle on `main` (snapshot context `4e4d75c4594b1072729ffe335c9bcea601749f71` was inspected in the third review; it is not this tree's pin): https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
- vLLM ModelOpt loader: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
- vLLM completion protocol: https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/
- vLLM batch invariance (beta, not assumed): https://docs.vllm.ai/en/latest/features/batch_invariance/
- FlashInfer issue 2723: https://github.com/flashinfer-ai/flashinfer/issues/2723
- SGLang issue 21802 (distinct gate/up weight scales dropped by a fusion; other model): https://github.com/sgl-project/sglang/issues/21802
- vLLM issue 53963 (SM120 sparse MLA rejects rope head dim 0): https://github.com/vllm-project/vllm/issues/53963
- NCCL graph collective participation: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html
- PyTorch CUDA graph memory pools: https://docs.pytorch.org/docs/stable/notes/cuda.html
- Compute Sanitizer: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html

Research, with their limits:

- arXiv:2609.15030, GLM-5.3-Flash hybrid-state restore. RedHat NVFP4, TP 4, not this port.
- arXiv:2603.08747, FP4 diagnosis on Qwen2.5. Probe placement only.
- arXiv:2512.02010, Four Over Six. A different recipe. Not used.

In-tree, at `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`:

- `src/tensorfold/cuda/nvfp4/checkpoint.cpp` `lane` — one weight, one alpha, scales `(K/64, mpad, 4)`
- `src/tensorfold/cuda/nvfp4/nvfp4q.cuh` `block_scale`
- `src/tensorfold/cuda/nvfp4/checkpoint.py` — `matmul_group`, `_out`
- `src/tensorfold/cuda/kernels/qmm.py` `split_k`
- `src/tensorfold/families/glm5_next/cuda/engine.py` — `MTP_DEFAULT`, dense window
- `src/tensorfold/families/glm5_next/cuda/forward.py` — `slots = top_k + 1`, prefill `ey` bf16
- `src/tensorfold/families/glm5_next/cuda/glue.py` — `swiglu`, `select`
- `src/tensorfold/families/glm5_next/cuda/graphs.py`
- `src/tensorfold/families/glm5_next/cuda/split.py`
- `src/tensorfold/cuda/geometry.py` — latent per rank, draft-head `9/16`
