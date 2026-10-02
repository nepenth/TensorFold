# GLM NVFP4 status

Branch: `plan/glm-nvidia-nvfp4`. Plan: `docs/plans/2026-10-02-glm-nvidia-nvfp4.md`. Updated 2026-10-02 after the CPU stretch.

Checkpoint: `nvidia/GLM-5.3-Flash-NVFP4` at `da920bb0b9f4a06727223a349e55468e38352348`. Reference: `BLOCKED_REFERENCE` (`nvidia-glm-nvfp4-reference.md`). No weight shard has been downloaded. No server has been stopped.

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

On this workstation, 47 tests passed and 2 were skipped: `load()` because PyTorch was not installed, and `split_device` because there was no CUDA device. `tests/cuda/` was not collected.

## Not implemented

- Device-indexed expert dispatch, pointer-table load, eager GPU oracles, packed prefill, CUDA graphs.
- Two-rank digest, missing-peer timeout, captured NCCL.
- `Fp4Linear.from_checkpoint` on a real projection. The codec is not `quant4`.
- vLLM record, teacher-forced compare, long context, speed.
- Recipe text. Nothing is `DENSE_FIDELITY_PASS`.

## Needs the two DGX Sparks

One Spark is enough for the skipped GPU tests, one dense projection, the pointer-table load, eager oracles, and the dispatch sanitizer.

Both Sparks are required for NCCL, graph replay across ranks, the recorded vLLM comparison, and any speed number. Do not pull the 190.4 GiB snapshot, and do not start `tensorfold serve`, until those single-Spark gates have passed and a window names how the existing serve is restored.

The first action on the cluster is to run the skipped tests and Task 5C–5D. It is not to serve the model.
