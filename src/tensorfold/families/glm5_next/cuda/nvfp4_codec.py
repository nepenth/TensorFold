"""Logical NVFP4 activation codec and the prepared-row layout. No CUDA.

The scale byte is chosen with TensorFold's own e4m3 decoder (``format.e4m3``). Current ModelOpt ``main`` clamps tiny
block scales toward ``2**-9`` before that cast, so a tiny block may not match this codec. Stored checkpoint weights
are never re-encoded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tensorfold.cuda.nvfp4.format import e4m3

MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CODE_LAYOUT = "u8-k-pairs-low-nibble-first"
SCALE_LAYOUT = "k64-mpad-4"   # scales[k_group, row, 4]; axis 0 is not the row


def e2m1_nibble(value: float) -> int:
    """Nearest E2M1 magnitude. A tie takes the larger magnitude. Zero stays unsigned."""

    if not math.isfinite(value):
        raise ValueError(f"nonfinite activation {value!r}")
    magnitude = min(abs(value), 6.0)
    choice = min(range(8), key=lambda i: (abs(magnitude - MAGNITUDES[i]), -MAGNITUDES[i]))
    sign = 8 if value < 0 and choice != 0 else 0
    return choice | sign


def _e4m3_byte(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError(f"nonfinite scale {value!r}")
    if value == 0:
        return 0
    table = e4m3(np.arange(256, dtype=np.uint8))
    finite = np.isfinite(table)
    errors = np.where(finite, np.abs(table - np.float32(value)), np.inf)
    return int(np.argmin(errors))


@dataclass
class PreparedRows:
    """One quantized activation. ``scales[0]`` is the first K group of every row, not row 0."""

    logical_rows: int
    logical_k: int
    padded_rows: int
    code_layout_id: str
    scale_layout_id: str
    codes: np.ndarray          # uint8 [padded_rows, K/2]
    scales: np.ndarray         # uint8 [K/64, padded_rows, 4]
    activation_factor: float
    owner: str

    def row(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.scale_layout_id != SCALE_LAYOUT or self.code_layout_id != CODE_LAYOUT:
            raise ValueError(f"unsupported layout {self.code_layout_id}/{self.scale_layout_id}")
        if not 0 <= index < self.logical_rows:
            raise ValueError(f"row {index} is outside {self.logical_rows}")
        return self.codes[index], self.scales[:, index, :]


def quantize_into(values: np.ndarray, activation_factor: float, codes: np.ndarray, scales: np.ndarray, *,
                  owner: str = "caller") -> PreparedRows:
    """Quantize ``values`` [rows, K] into the caller's ``codes`` and ``scales``. ``K`` is a multiple of 64."""

    if not math.isfinite(activation_factor) or activation_factor <= 0:
        raise ValueError(f"activation factor {activation_factor!r} must be finite and positive")
    if values.ndim != 2 or values.shape[1] % 64:
        raise ValueError(f"activations {getattr(values, 'shape', None)} need K a multiple of 64")
    rows, k = int(values.shape[0]), int(values.shape[1])
    padded = int(scales.shape[1]) if scales.ndim == 3 else 0
    if codes.shape != (padded, k // 2) or scales.shape != (k // 64, padded, 4):
        raise ValueError(f"codes {codes.shape} and scales {scales.shape} are not layout {SCALE_LAYOUT} for K {k}")
    if padded < rows or padded % 64:
        raise ValueError(f"padded rows {padded} must hold {rows} and be a multiple of 64")
    codes[:] = 0
    scales[:] = 0
    inverse = 1.0 / float(activation_factor)
    for row in range(rows):
        for block in range(k // 16):
            chunk = np.asarray(values[row, block * 16:(block + 1) * 16], dtype=np.float64)
            if not np.isfinite(chunk).all():
                raise ValueError("nonfinite activation")
            amax = float(np.max(np.abs(chunk))) if chunk.size else 0.0
            group, slot = divmod(block, 4)
            if amax == 0:
                scales[group, row, slot] = 0
                continue
            scale_byte = _e4m3_byte(inverse * amax / 6.0)
            scales[group, row, slot] = scale_byte
            decoded = float(e4m3(np.array([scale_byte], dtype=np.uint8))[0])
            mul = 0.0 if decoded == 0 else inverse / decoded
            for offset, item in enumerate(chunk):
                nibble = e2m1_nibble(float(item) * mul)
                pair = block * 8 + offset // 2
                if offset % 2 == 0:
                    codes[row, pair] = nibble
                else:
                    codes[row, pair] = int(codes[row, pair]) | (nibble << 4)
    return PreparedRows(rows, k, padded, CODE_LAYOUT, SCALE_LAYOUT, codes, scales, float(activation_factor), owner)
