# NVIDIA GLM NVFP4 reference feasibility

Status: `BLOCKED_REFERENCE`.

Checked 2026-10-02. The workstation had nothing on ports 8000 or 8080. The serving machines were inspected before the approved stop.

Inspected 2026-10-02 on the serving machines, read-only, before the approved stop. The process was up and still loading. Nothing was listening on port 8000, so no completion was recorded. The launch path was a RedHat compressed-tensors GLM-5.3-Flash NVFP4 tree, not `nvidia/GLM-5.3-Flash-NVFP4` at `da920bb0b9f4a06727223a349e55468e38352348`. A different checkpoint is not the oracle. The process was then stopped. Its weights were not deleted.

CPU loader and split work continues. `DENSE_FIDELITY_PASS` waits until an approved window can record that reference without a second copy of the weights resident.
