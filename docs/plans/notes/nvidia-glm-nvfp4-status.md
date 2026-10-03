# GLM NVFP4 status

Branch: `plan/glm-nvidia-nvfp4`. Plan: `docs/plans/2026-10-02-glm-nvidia-nvfp4.md`. Updated 2026-10-02 after the synthetic pointer-table pass.

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

On the CPU workstation, 47 tests passed and 2 were skipped: `load()` because PyTorch was not installed, and `split_device` because there was no CUDA device.

On one DGX Spark, PyTorch 2.13.0+cu130, capability (12, 1): those two tests passed, and `tests/cuda/test_glm_split_and_policy.py` collected and passed (10). The five NVFP4 CPU files passed in the same container (45). A later run packed one dense gate and ran the existing lane and prompt kernels. A synthetic routed layer then passed (14): E=4 owners match the address table by data pointer, and E=288 allocated pointer slots only. No model mount. Available memory stayed above 110 GiB.

## Not implemented

- Device-indexed expert dispatch, eager GPU oracles, packed prefill, CUDA graphs.
- A snapshot load. `load()` still raises `NVFP4 tensors are not wired`.
- Two-rank digest, missing-peer timeout, captured NCCL.
- A numeric envelope for the dense projection. The codec is still not `quant4`.
- vLLM record, teacher-forced compare, long context, speed.
- Recipe text. Nothing is `DENSE_FIDELITY_PASS`.

## Needs the two DGX Sparks

The skipped GPU tests have now passed on one Spark. A rank-local dense gate has been packed and run through the existing lane kernel and the prompt GEMM. A synthetic routed layer is a single-owner pointer table. Outputs of the dense gate were finite bf16. Recorded `split_k` is 2. No numeric envelope. No shard pull. No serve.

The next single-Spark gate is the eager oracles, then the dispatch sanitizer. A private continuation loop advances that batch if this session stops. It does not start a serve.

Both Sparks are required for NCCL, graph replay across ranks, the recorded comparison, and any speed number. Do not pull the 190.4 GiB snapshot, and do not start `tensorfold serve`, until those single-Spark gates have passed. The approved window already names how the previous serve is restored. That is not a license to start Task 12.
