# GLM NVFP4 status

Branch: `plan/glm-nvidia-nvfp4`. Plan: `docs/plans/2026-10-02-glm-nvidia-nvfp4.md`. Updated 2026-10-02 after the prompt and routed eager oracles.

Checkpoint: `nvidia/GLM-5.3-Flash-NVFP4` at `da920bb0b9f4a06727223a349e55468e38352348`. Reference: `BLOCKED_REFERENCE` (`nvidia-glm-nvfp4-reference.md`). No weight shard has been downloaded. The previous serve was a different checkpoint. It was stopped under an approved window. Restore steps are not in this public tree. The previous weights were left on disk.

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

On the CPU workstation, 47 tests passed and 2 were skipped: `load()` because PyTorch was not installed, and `split_device` because there was no CUDA device.

On one DGX Spark, PyTorch 2.13.0+cu130, capability (12, 1): those two tests passed, and `tests/cuda/test_glm_split_and_policy.py` collected and passed (10). The five NVFP4 CPU files passed in the same container (45). A later run packed one dense gate and ran the existing lane and prompt kernels. A synthetic routed layer then passed (14): E=4 owners match the address table by data pointer, and E=288 allocated pointer slots only. No model mount. Available memory stayed above 110 GiB.

## Not implemented

- CUDA graphs and a grouped device prefill GEMM. Task 10C packs on the host and reuses the lane eager oracle. That is not graph-qualified. One-token routed decode is indexed from the device table.
- A snapshot load. `load()` still raises `NVFP4 tensors are not wired`.
- Two-rank digest, missing-peer timeout, captured NCCL.
- A numeric envelope for the dense projection against a saved reference. The lane oracle uses `quant4`. That is not a reference match.
- vLLM record, teacher-forced compare, long context, speed.
- Recipe text. Nothing is `DENSE_FIDELITY_PASS`.

## Needs the two DGX Sparks

The skipped GPU tests have now passed on one Spark. A rank-local dense gate has been packed and run through the existing lane kernel and the prompt GEMM. A synthetic routed layer is a single-owner pointer table. Outputs of the dense gate were finite bf16. Recorded `split_k` is 2. No numeric envelope. No shard pull. No serve.

The dense lane eager oracle, the prompt eager oracle, and the routed eager oracle have passed on one Spark. Forty-six tests passed. Three two-GPU device-mismatch cases were skipped. Lane and prompt were not compared bitwise. A gate above 10 differs numerically from `mlp_prompt`. No numeric envelope. No shard pull. No serve.

The next remaining model gate is the full-forward graph and the approved snapshot. Task 10D's primitive passed on one Spark: a fixed expert bank, the existing lane kernel, and a device select. Supplied ids changed. Table addresses did not. Sequences were A then B then A, and graph then eager then graph. Prefill stays eager and is not graph-qualified. This is not a captured routed decode and not a full-forward graph. No snapshot pull. No serve.

Both Sparks are required for the recorded comparison and any speed number. Do not pull the 190.4 GiB snapshot, and do not start `tensorfold serve`, until the full-forward graph and the dry-run admission for the exact context exist. The approved window already names how the previous serve is restored. That is not a license to start Task 12.
