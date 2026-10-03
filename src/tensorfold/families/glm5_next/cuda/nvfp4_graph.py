"""Task 10D: supplied-id NVFP4 projection primitive on the existing lane kernel.

Prefill is eager and is not graph-qualified. No router or full forward is captured.
The eager table dispatch reads device slots on the host, so this primitive binds
immutable slots BEFORE capture, evaluates every expert with the existing lane,
and selects the requested results on device. Work scales with the expert bank;
this is primitive qualification, not production routed-decode qualification.
Weights are borrowed from retained owners, NEVER stacked or copied. Table storage
addresses AND slot contents MUST remain fixed for the lifetime of the graph.
"""

from __future__ import annotations

import math

import torch

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.families.glm5_next.cuda.nvfp4_table import RoutedTable

_TABLE_FIELDS = ("words_ptr", "bs_ptr", "n", "k", "alpha")


def _signature(tensor: torch.Tensor) -> tuple:
    return tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    a, b = left.untyped_storage(), right.untyped_storage()
    return a.data_ptr() < b.data_ptr() + b.nbytes() and b.data_ptr() < a.data_ptr() + a.nbytes()


class NVFP4PrimitiveGraph:
    """One prepared projection for decode batches of 1..6 rows.

    ``expert_ids`` is contiguous CUDA int64 ``(slots,)``; ``output`` is contiguous
    CUDA fp32/bf16 ``(slots, rows, n)``. Ids MUST be valid table indices. ``rows``
    comes from eager ``checkpoint.quant4`` under the projection's input factor.
    Setup may read table slots on the host; capture, eager launch and replay do
    not. Prepared rows may be updated in place between launches. Each instance
    owns its scratch and graph; launches on different streams MUST be ordered.
    """

    def __init__(self, rows: checkpoint.Rows4, table: RoutedTable, column: int,
                 expert_ids: torch.Tensor, output: torch.Tensor) -> None:
        if not checkpoint.available():
            raise RuntimeError("NVFP4 primitive graph requires an SM 12 CUDA device")
        if not isinstance(table, RoutedTable):
            raise TypeError("expected RoutedTable")
        if not isinstance(column, int) or not 0 <= column < 3:
            raise ValueError("projection column must be gate=0, up=1, or down=2")
        codes, scales = rows
        if (codes.ndim != 2 or not 1 <= codes.shape[0] <= 6 or codes.dtype != torch.uint8 or
                not codes.is_cuda or not codes.is_contiguous()):
            raise ValueError("prepared codes must be contiguous CUDA uint8 with 1..6 decode rows")
        device = codes.device
        if (expert_ids.ndim != 1 or expert_ids.numel() == 0 or expert_ids.dtype != torch.int64 or
                expert_ids.device != device or not expert_ids.is_contiguous()):
            raise ValueError("expert ids must be contiguous CUDA int64 (positive slots,)")
        fields = tuple(getattr(table, name) for name in _TABLE_FIELDS)
        experts = table.words_ptr.shape[0] if table.words_ptr.ndim == 2 else 0
        dtypes = (torch.int64, torch.int64, torch.int32, torch.int32, torch.float32)
        if experts == 0 or any(t.device != device or t.dtype != dtype or t.shape != (experts, 3) or
                               not t.is_contiguous() for t, dtype in zip(fields, dtypes)):
            raise ValueError("invalid CUDA pointer table fields")

        # Resolve only immutable table slots BEFORE capture. Owners keep every
        # borrowed packed buffer alive; no device slot read occurs in _launch.
        slots = tuple(t.detach().cpu().numpy()[:, column] for t in fields)
        owners = {(lin.words.data_ptr(), lin.bs.data_ptr()): lin for lin in table.linears}
        bindings = []
        for wp, bp, n, k, alpha in zip(*slots):
            lin = owners.get((int(wp), int(bp)))
            if lin is None or (lin.n, lin.k) != (int(n), int(k)):
                raise ValueError("table slots must refer to retained matching lane owners")
            if lin.words.device != device or lin.bs.device != device or not math.isfinite(float(alpha)):
                raise ValueError("lane owners must share the input device and have finite alpha")
            bindings.append((lin.words, lin.bs, float(alpha), lin.n, lin.k, lin.npad))
        n, k = bindings[0][3:5]
        if n <= 0 or k <= 0 or k % 64 or any(binding[3:5] != (n, k) for binding in bindings):
            raise ValueError("selected expert projections must have uniform positive n/K, K a multiple of 64")
        m = codes.shape[0]
        if codes.shape[1] != k // 2:
            raise ValueError("prepared codes width must match K/2")
        if (scales.device != device or scales.dtype != torch.uint8 or not scales.is_contiguous() or
                scales.ndim != 3 or scales.shape[0] != k // 64 or scales.shape[1] < m or
                scales.shape[1] % 64 or scales.shape[2] != 4):
            raise ValueError("prepared scales must be CUDA uint8 (K/64, mpad, 4)")
        if (output.device != device or output.dtype not in (torch.float32, torch.bfloat16) or
                output.shape != (expert_ids.numel(), m, n) or not output.is_contiguous()):
            raise ValueError("destination must be contiguous CUDA fp32/bf16 (slots, rows, n)")
        sources = (codes, scales, expert_ids, *fields, *(t for b in bindings for t in b[:2]))
        if any(_overlaps(output, source) for source in sources):
            raise ValueError("destination must not overlap inputs, ids, tables, or weights")

        from tensorfold.cuda.kernels import qmm

        self.rows, self.table = rows, table
        self.expert_ids, self.output = expert_ids, output
        self._bindings = tuple(bindings)
        self._bank = torch.empty((experts, m, n), dtype=output.dtype, device=device)
        self._destinations = tuple(self._bank[e] for e in range(experts))
        self._sk = qmm.split_k(n, k)
        self._tile = checkpoint.lane_tile(m, n, self._sk, checkpoint._sms(device.index))
        self._lane = checkpoint._ext().lane  # Build/load outside capture.
        self._signatures = tuple(_signature(t) for t in self._buffers())
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(device):
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                self._launch()
            current.wait_stream(stream)
            with torch.cuda.graph(self._graph, stream=stream):
                self._launch()
            current.wait_stream(stream)

    def _buffers(self) -> tuple[torch.Tensor, ...]:
        return (self.expert_ids, self.output, *self.rows, self._bank,
                *(getattr(self.table, name) for name in _TABLE_FIELDS),
                *(t for b in self._bindings for t in b[:2]))

    def _launch(self) -> None:
        for (words, scales, alpha, n, k, npad), destination in zip(self._bindings, self._destinations):
            self._lane(checkpoint.A4, *self.rows, words, scales, alpha, destination, None,
                       n, k, self._sk, npad, self._tile, self.output.dtype == torch.float32)
        torch.index_select(self._bank, 0, self.expert_ids, out=self.output)

    def _update(self, expert_ids: torch.Tensor) -> None:
        if tuple(_signature(t) for t in self._buffers()) != self._signatures:
            raise RuntimeError("captured primitive buffer addresses or metadata changed")
        if (expert_ids.device != self.expert_ids.device or expert_ids.dtype != self.expert_ids.dtype or
                expert_ids.shape != self.expert_ids.shape or not expert_ids.is_contiguous()):
            raise ValueError("expert ids must match captured CUDA device, dtype, shape and layout")
        self.expert_ids.copy_(expert_ids)

    def replay(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Copy supplied device ids into fixed storage and replay on the current stream."""
        with torch.cuda.device(self.output.device):
            self._update(expert_ids)
            self._graph.replay()
        return self.output

    def eager(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Launch the identical lane primitive without the graph, using the same buffers."""
        with torch.cuda.device(self.output.device):
            self._update(expert_ids)
            self._launch()
        return self.output
