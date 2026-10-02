# NVIDIA GLM NVFP4 reference feasibility

Status: `BLOCKED_REFERENCE`.

Checked 2026-10-02 from this workstation, read-only. Nothing was listening on `127.0.0.1` ports 8000 or 8080, and no other local model server was queried. No process was started or stopped.

The running service, if it is on the two Sparks, was not inspected. Its checkpoint, image digest, MoE backend, clamp, and cache dtype are unknown. A server that is not `nvidia/GLM-5.3-Flash-NVFP4` at revision `da920bb0b9f4a06727223a349e55468e38352348` is not the oracle.

CPU loader and split work continues. `DENSE_FIDELITY_PASS` waits until an approved window can record that reference without a second copy of the weights resident.
