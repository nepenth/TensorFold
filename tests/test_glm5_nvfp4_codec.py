"""Logical NVFP4 codec and prepared-row layout. These do not call the CUDA quantizer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfold.families.glm5_next.cuda.nvfp4_codec import e2m1_nibble, quantize_into


def _buffers(rows: int, k: int, padded: int):
    codes = np.zeros((padded, k // 2), np.uint8)
    scales = np.zeros((k // 64, padded, 4), np.uint8)
    return codes, scales


def test_zero_block_is_unsigned_zero_and_saturation_uses_the_largest_code():
    assert e2m1_nibble(0.0) == 0
    assert e2m1_nibble(-0.0) == 0
    assert e2m1_nibble(100.0) & 7 == 7
    assert e2m1_nibble(-100.0) & 7 == 7
    assert e2m1_nibble(-100.0) & 8 == 8


def test_a_midpoint_tie_takes_the_larger_magnitude():
    # 0.25 is equidistant from 0 and 0.5.
    assert e2m1_nibble(0.25) == e2m1_nibble(0.5)


def test_nonfinite_inputs_raise_and_a_non_unit_factor_changes_the_scale():
    with pytest.raises(ValueError, match="nonfinite"):
        e2m1_nibble(math.nan)
    values = np.zeros((1, 64), np.float32)
    values[0, :16] = 6.0
    unit_codes, unit_scales = _buffers(1, 64, 64)
    other_codes, other_scales = _buffers(1, 64, 64)
    quantize_into(values, 1.0, unit_codes, unit_scales)
    quantize_into(values, 2.0, other_codes, other_scales)
    assert unit_scales[0, 0, 0] != other_scales[0, 0, 0]


def test_scale_axis_zero_is_a_k_group_and_rows_do_not_share_it():
    values = np.zeros((2, 128), np.float32)
    values[0, :16] = 6.0
    values[1, 64:80] = 6.0
    codes, scales = _buffers(2, 128, 64)
    prepared = quantize_into(values, 1.0, codes, scales)
    assert prepared.scale_layout_id == "k64-mpad-4"
    assert int(prepared.scales[0, 0, 0]) != 0
    assert int(prepared.scales[0, 1, 0]) == 0
    assert int(prepared.scales[1, 1, 0]) != 0
    row_codes, row_scales = prepared.row(1)
    assert row_codes.shape == (64,)
    assert row_scales.shape == (2, 4)
    with pytest.raises(ValueError, match="unsupported layout"):
        prepared.code_layout_id = "row-major"
        prepared.row(0)


def test_quantize_into_rejects_a_short_buffer_and_does_not_require_a_new_allocation():
    values = np.ones((1, 64), np.float32)
    codes, scales = _buffers(1, 64, 64)
    before = codes.__array_interface__["data"][0]
    quantize_into(values, 1.0, codes, scales, owner="caller")
    assert codes.__array_interface__["data"][0] == before
    with pytest.raises(ValueError, match="padded rows"):
        quantize_into(np.ones((65, 64), np.float32), 1.0, codes, scales)
    with pytest.raises(ValueError, match="finite and positive"):
        quantize_into(values, 0.0, codes, scales)
