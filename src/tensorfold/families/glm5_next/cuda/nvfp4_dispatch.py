"""One eager lane projection selected from a RoutedTable's CUDA slots."""

from __future__ import annotations

import torch

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.families.glm5_next.cuda.nvfp4_table import RoutedTable

GATE, UP, DOWN = 0, 1, 2


def projection(rows: checkpoint.Rows4, table: RoutedTable, expert_id: int, column: int,
               out: torch.Tensor) -> torch.Tensor:
    """Write one projection to caller-owned contiguous fp32/bf16 ``out``.

    Quantize with ``checkpoint.quant4(x, act)`` before calling: the five table
    fields carry the output alpha, not the input quantization factor. The table
    MUST keep its owners alive through CUDA completion. Slot reads synchronize;
    this Task 10A entry point is for eager execution.
    """

    if table.words_ptr.ndim != 2 or table.words_ptr.shape[1] != 3:
        raise ValueError("tables must have shape (experts, 3)")
    if not isinstance(expert_id, int) or not 0 <= expert_id < table.words_ptr.shape[0]:
        raise ValueError("expert id is out of range")
    if not isinstance(column, int) or not GATE <= column <= DOWN:
        raise ValueError("projection column must be gate=0, up=1, or down=2")
    codes, scales = rows
    if codes.ndim != 2 or codes.shape[0] <= 0 or not codes.is_cuda or codes.dtype != torch.uint8:
        raise ValueError("codes must be CUDA uint8 (positive rows, K/2)")
    fields = ((table.words_ptr, torch.int64), (table.bs_ptr, torch.int64),
              (table.n, torch.int32), (table.k, torch.int32), (table.alpha, torch.float32))
    shape = (table.words_ptr.shape[0], 3)
    if any(t.dtype != dtype or t.device != codes.device or not t.is_cuda or
           tuple(t.shape) != shape or not t.is_contiguous() for t, dtype in fields):
        raise ValueError("tables must be contiguous CUDA (experts, 3) with int64 pointers, int32 n/k, fp32 alpha")
    n, k = int(table.n[expert_id, column]), int(table.k[expert_id, column])
    if n <= 0 or k <= 0 or k % 64:
        raise ValueError("n/K must be positive, K a multiple of 64")
    if out.ndim != 2 or out.shape != (codes.shape[0], n):
        raise ValueError("destination must have shape (rows, n)")
    if out.dtype not in (torch.float32, torch.bfloat16) or out.device != codes.device:
        raise ValueError("destination must be fp32 or bf16 on the input device")
    if not out.is_contiguous():
        raise ValueError("destination must be contiguous and have no internal overlap")

    from tensorfold.cuda.kernels import qmm

    sk = qmm.split_k(n, k)
    tile = checkpoint.lane_tile(codes.shape[0], n, sk, checkpoint._sms(codes.device.index))
    checkpoint._ext().dispatch(codes, scales, table.words_ptr, table.bs_ptr, table.n, table.k, table.alpha,
                               expert_id, column, out, sk, tile)
    return out
