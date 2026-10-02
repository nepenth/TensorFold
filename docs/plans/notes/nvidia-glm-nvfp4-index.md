# NVIDIA GLM-5.3-Flash NVFP4 index census

Checked 2026-10-02 against revision `da920bb0b9f4a06727223a349e55468e38352348` (see the manifest). The index was downloaded. One safetensors header was range-read (`model-00001-of-00033.safetensors`, 508,008 header bytes). The response was the header, not the shard body.

| Fact | Value |
| --- | --- |
| Tensors | 147,661 |
| Shards | 33 |
| `metadata.total_size` | 204,419,110,596 |
| `weight_packed` | 0 |
| `weight_global_scale` | 0 |
| `mlp.experts` | 146,016 |
| Layer 45 tensors | 889, none with `scale` in the name |

`layers.3.mlp.experts.0.gate_proj.weight` is `U8 [2048, 2048]`. Its `weight_scale` is `F8_E4M3 [2048, 256]`. Its `input_scale` is `F32 []`.

These match the plan's census table. Loader work can use that table.
