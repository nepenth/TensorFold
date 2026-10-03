# GLM NVFP4 status

Branch: `plan/glm-nvidia-nvfp4`. Plan: `docs/plans/2026-10-02-glm-nvidia-nvfp4.md`. Updated 2026-10-03 after one measured layer and the incremental host reserve.

Checkpoint: `nvidia/GLM-5.3-Flash-NVFP4` at `da920bb0b9f4a06727223a349e55468e38352348`. Reference: `BLOCKED_REFERENCE` (`nvidia-glm-nvfp4-reference.md`). The pinned snapshot is local on both ranks. The previous serve was a different checkpoint. It was stopped under an approved window. Restore steps are not in this public tree.

## Implemented

| Commit | What it is |
| --- | --- |
| `5d7963d` | Manifest and numeric policy. |
| `7e127ae` | Offline comparator. Top-k logprobs are not a full-vocabulary pass. An inconclusive margin is not a near-tie. |
| `cefba59` | CPU startup policy. Unset MTP is not treated as `"1"` for this quant. |
| `8c7a72a` | Census at the pinned revision. 147,661 tensors, 33 shards, no `weight_packed`, layer 45 unscaled. |
| `4e41131` | Reference marked blocked. |
| `daf2a03` | Family gate admits static NVFP4 group 16 only. `load()` still raises `NVFP4 tensors are not wired`. Explicit MTP, a drafter, and `--parallel` fail before the engine import. Modelopt forces `mtp_on` false and drops the draft-head byte term. |
| `90ce7a8` | CPU splits. Scalars replicate. Packed width 96 is rejected. Rank-folder provenance is checked. |
| `0203859` | Logical activation codec, prepared-row layout (`k64-mpad-4`), and `nvfp4_inventory`. |
| `585faa1` | Dense lane eager oracle. Common-input quant, `glue.swiglu`, one down into the caller fp32 buffer. Sixteen Spark tests. Not a reference envelope. |
| `4383001` | Prompt eager oracle and routed eager oracle. Lane and prompt are not compared bitwise. Each down reads its own intermediate. Shared BF16 MLP still contributes when every routed weight is zero. |
| `50ddfcf` | One device-indexed projection. 64-bit table address, existing lane kernel, 66 Spark tests. Sanitizer zero errors on the canary only. |
| `b1b583e` | One-token routed decode from the device table. Eight experts, distinct intermediate quants, shared BF16 branch, ordered combine. 29 Spark tests. |
| `9a709ee` | Prefill pack and slot-ordered combine. Lane eager oracle, not a grouped device GEMM, not graph-qualified. 59 Spark tests. |
| `99fe063` | Two-rank partials on one device. Row-split concat, column-split fp32 sum, rank 0 first. `split_k` recorded. No frozen envelope. 17 Spark tests. |
| `cbd6603` | Two-rank NCCL smoke. Eager and captured ordered sum, two replays. Missing store peer failed in 3 seconds. Revision mismatch and independent expert ids recorded on both ranks. |
| `8a28826` | Supplied-id primitive graph. Fixed bank, existing lane, device select. 79 Spark tests. Prefill is not graph-qualified. Not a full-forward graph. |
| `dec9b8f` | Fixture load. Routed experts and a dense layer with scales pack through `Fp4Linear.from_checkpoint`. Layer 45 names in the index are skipped. A draft head still raises. 37 Spark tests. Not a snapshot load. |
| `2935fe8` | Packed-layer forward. One token matches `routed_decode` on the same table. Dense scales use `dense_eager`. Multi-row stays eager and is not graph-qualified. Grouped experts are not called. 20 Spark tests. Not a snapshot load. Not a serve. |
| `cdbee00` | Synthetic engine step. Both ranks, one token, finite logits. Grouped experts are not called. No CUDA graph. Fixture hidden size is 128. `hc_mult` and the convolution width match the pinned config. Not a snapshot load. |
| `51f420e` | Per-expert global scales. Each expert's `weight_scale_2` and `input_scale` land in its alpha slot. Same-expert shard disagreement still raises. 35 Spark tests. One language-model MoE layer packed: 288 experts, 864 linears. Not a full snapshot load. Not a serve. |
| `159d73e` | Header census and a per-layer host reserve. One rank keeps 89.200 GiB. Peak 92.632 GiB at layer 43 if the packed layer overlaps the current reader's raw spans. Four CPU tests. No resident load. No serve. |
| `df46d92` | The reserve check uses the layer increment, not the running peak. Four CPU tests. No resident load. No serve. |

On the CPU workstation, 47 tests passed and 2 were skipped: `load()` because PyTorch was not installed, and `split_device` because there was no CUDA device.

On one DGX Spark, PyTorch 2.13.0+cu130, capability (12, 1): those two tests passed, and `tests/cuda/test_glm_split_and_policy.py` collected and passed (10). The five NVFP4 CPU files passed in the same container (45). A later run packed one dense gate and ran the existing lane and prompt kernels. A synthetic routed layer then passed (14): E=4 owners match the address table by data pointer, and E=288 allocated pointer slots only. No model mount. Available memory stayed above 110 GiB.

## Not implemented

- CUDA graphs and a grouped device prefill GEMM. Task 10C packs on the host and reuses the lane eager oracle. That is not graph-qualified. One-token routed decode is indexed from the device table.
- A resident snapshot load. The header census did not allocate the full rank. One rank keeps 95,777,735,492 bytes (89.200 GiB). The running peak is 99,462,461,744 bytes (92.632 GiB) at layer 43 if the packed layer overlaps the current reader's raw spans of that layer and the next. One MoE layer, then a second, were packed and dropped: each retained 2,063,597,568 bytes (1.921875 GiB) and peaked at 4,107,894,784 bytes allocated. The second drop returned host available to the same level. Forty-two such tables are 80.71875 GiB. The reserve check now asks for that layer's increment, not the running peak, because available memory no longer includes bytes already resident. A cgroup cap is not that check. The full load has not started. Modelopt does not capture a CUDA graph.
- Two-rank digest, missing-peer timeout, captured NCCL.
- A numeric envelope for the dense projection against a saved reference. The lane oracle uses `quant4`. That is not a reference match.
- vLLM record, teacher-forced compare, long context, speed.
- Recipe text. Nothing is `DENSE_FIDELITY_PASS`.

## Needs the two DGX Sparks

The skipped GPU tests have now passed on one Spark. A rank-local dense gate has been packed and run through the existing lane kernel and the prompt GEMM. A synthetic routed layer is a single-owner pointer table. Outputs of the dense gate were finite bf16. Recorded `split_k` is 2. No numeric envelope. No shard pull. No serve.

The dense lane eager oracle, the prompt eager oracle, and the routed eager oracle have passed on one Spark. Forty-six tests passed. Three two-GPU device-mismatch cases were skipped. Lane and prompt were not compared bitwise. A gate above 10 differs numerically from `mlp_prompt`. No numeric envelope. No shard pull. No serve.

The pinned snapshot is local on both ranks: 33 shards, index hash `26765b2601fd246ef361cfb9f5e10f9fb291a59e05ad0a109062f3a4747c7fd1`, config hash `41db2811023b40ba4c8f8bbba88bce7dff377af51ecd18b32469a2a07064ebaf`. The archive config omitted the layer-45 ignore lines; the pinned config replaced it after the archive byte count matched. No serve. `spark-llm` is not advertised. Layers 3 through 44 each packed and were dropped before the next: 288 experts and 864 linears every layer. Dense layers 0, 1, and 2 each packed as one projection triple and were dropped. Layer 45 was not loaded. This is not a resident snapshot. Modelopt does not capture a CUDA graph.

Both Sparks are required for the recorded comparison and any speed number. Do not start `tensorfold serve` until the dry-run admission for the exact context exists. A serve in this window must not advertise `spark-llm`. Add that alias back only after the window is done, on the restored profile.

Header census read safetensors headers only. Index hash `26765b2601fd246ef361cfb9f5e10f9fb291a59e05ad0a109062f3a4747c7fd1`, 147,661 tensors, 33 shards. One rank keeps 95,777,735,492 bytes (89.200 GiB), including the replicated head. The running peak is 99,462,461,744 bytes (92.632 GiB) at layer 43. That peak counts the packed layer plus the current reader's raw spans of layers 43 and 44. The column spans still include the other rank's bytes. The retained set does not. Four CPU tests passed. No resident load. No serve.
