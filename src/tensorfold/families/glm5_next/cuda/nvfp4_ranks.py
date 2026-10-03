"""Tasks 9B/9C: two-rank agreement and ordered fp32 NCCL partials.

This seam owns no weights and launches no expert kernels. Startup checks run
outside capture; ordered_sum uses the existing current-stream all_gather in
both eager execution and CUDA graphs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch

from tensorfold.cuda.comm import NCCL

_DIGEST_WORDS = 4096
_SHARED_FIELDS = ("revision", "world", "context", "kernel_policy", "cache_mode")


@dataclass(frozen=True)
class RankDigest:
    revision: str
    world: int
    rank: int
    context: int
    kernel_policy: str
    cache_mode: str


def validate_digests(local: RankDigest, gathered: tuple[RankDigest, ...]) -> None:
    """Compare shared settings and validate each rank's own identity."""
    if local.world != 2:
        raise ValueError("rank digest mismatch: world (expected two ranks)")
    if local.rank not in (0, 1):
        raise ValueError("peer rank must be the other rank (expected rank 0 or 1)")
    if len(gathered) != 2:
        raise ValueError("expected two rank digests")
    for rank, record in enumerate(gathered):
        if record.rank != rank:
            raise ValueError("peer rank must be the other rank (rank-order digest records)")
        for field in _SHARED_FIELDS:
            if getattr(record, field) != getattr(local, field):
                raise ValueError(f"rank digest mismatch: {field}")


def _encode_digest(digest: RankDigest, device: torch.device) -> torch.Tensor:
    payload = json.dumps(asdict(digest), sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) >= _DIGEST_WORDS:
        raise ValueError("rank digest is too large")
    # int32 is already supported by NCCL; word 0 stores the UTF-8 byte count.
    record = torch.zeros(_DIGEST_WORDS, dtype=torch.int32, device=device)
    record[:len(payload) + 1] = torch.tensor([len(payload), *payload], dtype=torch.int32, device=device)
    return record


def _decode_digest(record: torch.Tensor) -> RankDigest:
    words = record.tolist()
    size = words[0]
    if not 0 < size < _DIGEST_WORDS:
        raise ValueError("invalid rank digest length")
    return RankDigest(**json.loads(bytes(words[1:size + 1])))


class NVFP4Ranks:
    """Two-rank test seam using tensorfold.cuda.comm.NCCL, or a fake comm."""

    def __init__(self, comm: NCCL, digest: RankDigest, *, device: torch.device | str | None = None):
        if comm.world != 2 or comm.rank not in (0, 1):
            raise ValueError("NVFP4 rank checks require exactly two ranks")
        self.comm, self.digest = comm, digest
        self.device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.expert_records: tuple[torch.Tensor, ...] | None = None

    def check_digest(self) -> tuple[RankDigest, ...]:
        local = _encode_digest(self.digest, self.device)
        gathered = torch.empty((2, local.numel()), dtype=local.dtype, device=self.device)
        self.comm.all_gather(local, gathered)
        digests = tuple(_decode_digest(record) for record in gathered.cpu())
        validate_digests(self.digest, digests)
        return digests

    def check_expert_ids(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Snapshot this rank's eight ids, gather both records, then compare.

        Both records remain available on a mismatch. NEVER replace local ids
        with the peer's selections.
        """
        if expert_ids.numel() != 8 or expert_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("expected eight integer expert ids")
        if expert_ids.device != self.device:
            raise ValueError("expert ids must be on the rank device")
        local = expert_ids.detach().clone().contiguous().reshape(8)
        gathered = torch.empty((2, 8), dtype=local.dtype, device=local.device)
        self.comm.all_gather(local, gathered)
        # Retain independent CPU records even when comparison raises.
        self.expert_records = tuple(gathered.cpu().unbind(0))
        if not torch.equal(*self.expert_records):
            raise ValueError("expert ids differ between ranks")
        return local

    def ordered_sum(self, partial: torch.Tensor, gathered: torch.Tensor | None = None,
                    output: torch.Tensor | None = None) -> torch.Tensor:
        """All-gather fp32 partials, then copy rank 0 and add rank 1.

        Caller-owned scratch/output can be reused for capture and replay.
        """
        if partial.dtype != torch.float32 or partial.device != self.device or not partial.is_contiguous():
            raise ValueError("partials must be contiguous fp32 on the rank device")
        shape = (2, *partial.shape)
        if gathered is None:
            gathered = torch.empty(shape, dtype=partial.dtype, device=partial.device)
        if output is None:
            output = torch.empty_like(partial)
        for tensor, expected_shape in ((gathered, shape), (output, partial.shape)):
            if (tensor.shape != expected_shape or tensor.dtype != partial.dtype or tensor.device != partial.device
                    or not tensor.is_contiguous()):
                raise ValueError("ordered sum scratch/output must match fp32 partial shape and device")
        self.comm.all_gather(partial, gathered)
        output.copy_(gathered[0])
        output.add_(gathered[1])
        return output
