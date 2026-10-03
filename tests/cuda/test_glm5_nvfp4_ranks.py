"""Tasks 9B/9C: CPU fake comms and opt-in bounded two-GPU NCCL tests."""

# ruff: noqa: E402 -- torch is optional for CPU-only installs.

import inspect
import json
import multiprocessing
import os
from pathlib import Path
import socket
import time
from dataclasses import replace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import comm as comm_module
from tensorfold.cuda.comm import NCCL
from tensorfold.families.glm5_next.cuda.nvfp4_ranks import (
    NVFP4Ranks,
    RankDigest,
    _encode_digest,
)


class FakeComm:
    """Only implements the existing NCCL send, recv interface; no broadcast."""

    world = 2

    def __init__(self, rank, records):
        self.rank = rank
        self.records = tuple(record.clone() for record in records)
        self.inputs = []

    def all_gather(self, send, recv):
        assert send.is_contiguous() and recv.is_contiguous()
        assert recv.numel() == 2 * send.numel() and recv.dtype == send.dtype
        self.inputs.append(send.clone())
        for target, record in zip(recv, self.records):
            target.copy_(record)


def _digest(rank=0):
    return RankDigest("test-revision", 2, rank, 4096, "test-policy", "test-cache")


def _fake_digest_check(rank, digests):
    records = [_encode_digest(digest, torch.device("cpu")) for digest in digests]
    comm = FakeComm(rank, records)
    return NVFP4Ranks(comm, _digest(rank), device="cpu"), comm


_MISMATCHES = [("revision", "different"), ("world", 3), ("context", 8192),
               ("kernel_policy", "different"), ("cache_mode", "different")]


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("field,value", _MISMATCHES)
def test_digest_mismatch_fake_comm(rank, field, value):
    digests = [_digest(0), _digest(1)]
    digests[1 - rank] = replace(digests[1 - rank], **{field: value})
    ranks, comm = _fake_digest_check(rank, digests)
    with pytest.raises(ValueError, match=field):
        ranks.check_digest()
    assert len(comm.inputs) == 1
    assert torch.equal(comm.inputs[0], _encode_digest(_digest(rank), torch.device("cpu")))


@pytest.mark.parametrize("rank", (0, 1))
def test_digest_rank_differs_and_peer_must_be_other_rank(rank):
    ranks, comm = _fake_digest_check(rank, (_digest(0), _digest(1)))
    assert ranks.check_digest() == (_digest(0), _digest(1))
    assert not torch.equal(*comm.records)
    digests = [_digest(0), _digest(1)]
    digests[1 - rank] = replace(digests[1 - rank], rank=rank)
    ranks, _ = _fake_digest_check(rank, digests)
    with pytest.raises(ValueError, match="peer rank"):
        ranks.check_digest()


@pytest.mark.parametrize("rank", (0, 1))
def test_local_world_mismatch_is_exchanged_before_raising(rank):
    digests = [_digest(0), _digest(1)]
    digests[rank] = replace(digests[rank], world=3)
    records = [_encode_digest(digest, torch.device("cpu")) for digest in digests]
    comm = FakeComm(rank, records)
    ranks = NVFP4Ranks(comm, digests[rank], device="cpu")
    with pytest.raises(ValueError, match="world"):
        ranks.check_digest()
    assert len(comm.inputs) == 1
    assert torch.equal(comm.inputs[0], records[rank])


@pytest.mark.parametrize("matching", (True, False))
@pytest.mark.parametrize("dtype", (torch.int32, torch.int64))
def test_expert_ids_are_own_records_without_broadcast(matching, dtype):
    first = torch.tensor([7, 1, 0, 6, 2, 5, 3, 4], dtype=dtype)
    second = first.clone() if matching else first.flip(0)
    for rank, own in enumerate((first, second)):
        before = own.clone()
        comm = FakeComm(rank, (first, second))
        ranks = NVFP4Ranks(comm, _digest(rank), device="cpu")
        if matching:
            snapshot = ranks.check_expert_ids(own)
            assert snapshot.data_ptr() != own.data_ptr()
            assert torch.equal(snapshot, own)
        else:
            with pytest.raises(ValueError, match="expert ids"):
                ranks.check_expert_ids(own)
        assert torch.equal(comm.inputs[0], before)
        assert torch.equal(own, before)
        assert all(torch.equal(actual, expected)
                   for actual, expected in zip(ranks.expert_records, (first, second)))


def _partials(step):
    # Cancellation, fp32 values that would change under bf16 rounding, signed
    # zero, and asymmetric rank data. Ordered records also expose reversed ranks.
    return (torch.tensor([1.00390625, 1e8, -0.0, 0.125 + step]),
            torch.tensor([1.0039065, -1e8, 0.0, 2.5 + 2 * step]))


def test_ordered_fp32_sum_fake_comm(monkeypatch):
    records = _partials(0)
    comm = FakeComm(1, records)
    ranks = NVFP4Ranks(comm, _digest(1), device="cpu")
    gathered, output = torch.empty((2, 4)), torch.empty(4)
    calls = []
    copy, add = torch.Tensor.copy_, torch.Tensor.add_

    def traced_copy(self, source, *args, **kwargs):
        if self is output:
            calls.append(("copy", source.data_ptr()))
        return copy(self, source, *args, **kwargs)

    def traced_add(self, source, *args, **kwargs):
        if self is output:
            calls.append(("add", source.data_ptr()))
        return add(self, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", traced_copy)
    monkeypatch.setattr(torch.Tensor, "add_", traced_add)
    assert ranks.ordered_sum(records[1], gathered, output) is output
    assert calls == [("copy", gathered[0].data_ptr()), ("add", gathered[1].data_ptr())]
    assert torch.equal(gathered, torch.stack(records))
    assert torch.equal(output.view(torch.int32), (records[0] + records[1]).view(torch.int32))
    rounded = records[0].bfloat16().float() + records[1].bfloat16().float()
    assert not torch.equal(output, rounded)


def test_constructor_timeout_defaults_to_600_and_reaches_store(monkeypatch):
    import torch.distributed as distributed

    assert inspect.signature(NCCL).parameters["timeout"].default == 600
    monkeypatch.setattr(comm_module, "_library", Mock(return_value=Mock()))
    observed = []

    def missing_peer(*args, **kwargs):
        observed.append(kwargs["timeout"].total_seconds())
        raise RuntimeError("missing peer timeout")

    monkeypatch.setattr(distributed, "TCPStore", missing_peer)
    for kwargs in ({}, {"timeout": 2.0}):
        with pytest.raises(RuntimeError, match="missing peer timeout"):
            NCCL(0, 2, "127.0.0.1", 12345, **kwargs)
    assert observed == [600, 2]


@pytest.fixture
def two_gpus():
    if os.environ.get("TENSORFOLD_TEST_NCCL") != "1":
        pytest.skip("set TENSORFOLD_TEST_NCCL=1 for two-rank NCCL tests")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_workers(tmp_path, case, *, ranks=(0, 1), field=None, deadline_seconds=90):
    ctx = multiprocessing.get_context("spawn")
    port = _port()
    processes = []
    try:
        for rank in ranks:
            process = ctx.Process(target=_nccl_worker, args=(rank, port, str(tmp_path), case, field))
            process.start()
            processes.append(process)
        deadline = time.monotonic() + deadline_seconds
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert all(not process.is_alive() for process in processes), "NCCL worker exceeded deadline"
        assert all(process.exitcode == 0 for process in processes), "NCCL worker failed"
        return [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in ranks]
    finally:
        # A failed assertion or hung collective MUST leave no live process.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            assert not process.is_alive()
            process.close()


def _nccl_worker(rank, port, directory, case, field, master="127.0.0.1", device=None, timeout=10.0):
    torch.cuda.set_device(rank if device is None else device)
    result = {"rank": rank}
    if case == "missing":
        started = time.monotonic()
        try:
            NCCL(rank, 2, master, port, timeout=2)
        except Exception as exc:
            elapsed = time.monotonic() - started
            assert "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            assert elapsed < 10, f"missing peer took {elapsed:.2f} seconds"
            result.update(error=str(exc), elapsed=elapsed)
        else:
            raise AssertionError("missing peer did not raise")
    else:
        comm = NCCL(rank, 2, master, port, timeout=timeout)
        digest = _digest(rank)
        if case == "digest" and rank == 1:
            digest = replace(digest, **{field: dict(_MISMATCHES)[field]})
        ranks = NVFP4Ranks(comm, digest)
        if case == "digest":
            try:
                ranks.check_digest()
            except ValueError as exc:
                assert field in str(exc)
                result["error"] = str(exc)
            else:
                raise AssertionError("digest mismatch did not raise")
        else:
            digests = ranks.check_digest()
            result["digest_ranks"] = [digest.rank for digest in digests]
            # Construct each rank's own ids from its own input; no rank supplies
            # another rank's selections.
            own_input = torch.tensor([7, 1, 0, 6, 2, 5, 3, 4], dtype=torch.int64, device=ranks.device)
            if case == "ids" and rank == 1:
                own_input = own_input.flip(0)
            before = own_input.clone()
            if case == "ids":
                try:
                    ranks.check_expert_ids(own_input)
                except ValueError as exc:
                    assert "expert ids" in str(exc)
                    result["error"] = str(exc)
                else:
                    raise AssertionError("independent expert mismatch did not raise")
            else:
                ranks.check_expert_ids(own_input)
            assert torch.equal(own_input, before)
            result["own_ids"] = own_input.cpu().tolist()
            result["expert_records"] = [record.tolist() for record in ranks.expert_records]
            if case == "sum":
                result.update(_eager_and_graph(ranks))
        # CPU store synchronization keeps both workers alive until all native
        # NCCL operations have completed; process exit releases the communicator.
        # Record the outcome before the final barrier. A peer that exits after
        # its own record must not erase this side's evidence.
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(result))
        comm.ready("finished", every=0.1, timeout=10)
        return
    Path(directory, f"rank-{rank}.json").write_text(json.dumps(result))


def _eager_and_graph(ranks):
    rank, comm = ranks.digest.rank, ranks.comm
    partial = torch.empty(4, dtype=torch.float32, device=ranks.device)
    gathered = torch.empty((2, 4), dtype=torch.float32, device=ranks.device)
    output = torch.empty_like(partial)
    results = []

    def load(step):
        records = _partials(step)
        partial.copy_(records[rank])
        return torch.stack(records), records[0] + records[1]

    def check(expected_records, expected_sum):
        torch.cuda.synchronize()
        actual_records, actual_sum = gathered.cpu(), output.cpu()
        assert torch.equal(actual_records.view(torch.int32), expected_records.view(torch.int32))
        assert torch.equal(actual_sum.view(torch.int32), expected_sum.view(torch.int32))
        results.append(actual_sum.view(torch.int32).tolist())

    for step in range(2):
        expected_records, expected_sum = load(step)
        ranks.ordered_sum(partial, gathered, output)
        check(expected_records, expected_sum)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ranks.ordered_sum(partial, gathered, output)
    stream.synchronize()
    comm.ready("capture", every=0.1, timeout=10)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ranks.ordered_sum(partial, gathered, output)
    comm.ready("captured", every=0.1, timeout=10)
    for step in range(2):
        expected_records, expected_sum = load(step)
        # Poison the destinations so a replay MUST execute the captured gather
        # and sum, rather than passing with stale eager output.
        gathered.fill_(float("nan"))
        output.fill_(float("nan"))
        comm.ready(f"replay-{step}", every=0.1, timeout=10)
        graph.replay()
        check(expected_records, expected_sum)
    assert results[:2] == results[2:]
    return {"sum_bits": results, "rank_order": [0, 1], "replays": 2}


def test_two_rank_eager_and_captured_ordered_partials(two_gpus, tmp_path):
    results = _run_workers(tmp_path, "sum")
    assert [result["rank"] for result in results] == [0, 1]
    assert results[0]["sum_bits"] == results[1]["sum_bits"]
    for result in results:
        assert result["digest_ranks"] == result["rank_order"] == [0, 1]
        assert result["replays"] == 2
        assert result["own_ids"] == result["expert_records"][result["rank"]]
        assert result["expert_records"][0] == result["expert_records"][1]


@pytest.mark.parametrize("field,value", _MISMATCHES)
def test_two_rank_digest_mismatch_exits(two_gpus, tmp_path, field, value):
    results = _run_workers(tmp_path, "digest", field=field)
    assert all(field in result["error"] for result in results)


def test_two_rank_expert_mismatch_exits_without_broadcast(two_gpus, tmp_path):
    results = _run_workers(tmp_path, "ids")
    assert results[0]["own_ids"] != results[1]["own_ids"]
    assert results[0]["expert_records"] == results[1]["expert_records"]
    for result in results:
        assert result["own_ids"] == result["expert_records"][result["rank"]]
        assert "expert ids" in result["error"]


@pytest.mark.parametrize("rank", (0, 1))
def test_missing_peer_raises_and_leaves_no_process(two_gpus, tmp_path, rank):
    results = _run_workers(tmp_path, "missing", ranks=(rank,), deadline_seconds=20)
    assert results[0]["elapsed"] < 10


if __name__ == "__main__":
    # Cross-host runner. The master address stays in the environment, not in this file.
    _nccl_worker(
        int(os.environ["TF_NCCL_RANK"]),
        int(os.environ["TF_NCCL_PORT"]),
        os.environ["TF_NCCL_OUT"],
        os.environ["TF_NCCL_CASE"],
        os.environ.get("TF_NCCL_FIELD") or None,
        master=os.environ["TF_NCCL_MASTER"],
        device=int(os.environ.get("TF_NCCL_DEVICE", "0")),
        timeout=float(os.environ.get("TF_NCCL_TIMEOUT", "60")),
    )
