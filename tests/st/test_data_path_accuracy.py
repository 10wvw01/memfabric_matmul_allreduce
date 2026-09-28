from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import torch
import torch_npu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "common"))
from mfmc2_ctypes import Mfmc2Api  # noqa: E402

K = N = 2048
NZ_FORMAT = 29


def _nonzero_pair(logical_rank: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(2026092807 + logical_rank * 100003 + m)
    x = torch.randn((m, K), generator=g, dtype=torch.float32).half()
    w = torch.randn((N, K), generator=g, dtype=torch.float32).mul_(0.02).half()
    return x, w


def _rank_inputs(rank: int, source_rank: int, m: int):
    x_cpu, w_cpu = _nonzero_pair(source_rank, m)
    if rank != source_rank:
        # A zero activation makes this rank's local partial exactly zero while
        # preserving the same valid weight layout/shape contract.
        x_cpu = torch.zeros_like(x_cpu)
    x = x_cpu.npu()
    w = w_cpu.npu()
    return x, torch_npu.npu_format_cast(w, NZ_FORMAT)


def _source_reference(source_rank: int, m: int) -> torch.Tensor:
    # Build the reference on the current NPU using the exact stock 310P NZ
    # linear path. Both processes independently construct the same reference.
    x_cpu, w_cpu = _nonzero_pair(source_rank, m)
    x = x_cpu.npu()
    w_nz = torch_npu.npu_format_cast(w_cpu.npu(), NZ_FORMAT)
    return torch.nn.functional.linear(x, w_nz)


def _run_rank(rank: int, q: int, shapes: list[int], barrier: mp.Barrier,
              result_q: mp.Queue) -> None:
    try:
        torch.npu.set_device(rank)
        api = Mfmc2Api()
        stream = api.create_stream()
        try:
            for m in shapes:
                for source_rank in (0, 1):
                    x, w_nz = _rank_inputs(rank, source_rank, m)
                    expected = _source_reference(source_rank, m)
                    output = torch.empty(
                        (m, N), dtype=torch.float16, device=f"npu:{rank}"
                    )
                    torch.npu.synchronize()
                    barrier.wait()
                    api.launch(
                        x.data_ptr(),
                        w_nz.data_ptr(),
                        m,
                        rank,
                        q,
                        output.data_ptr(),
                        stream,
                    )
                    barrier.wait()

                    # source_rank: validates local MM/send-arena + reduction
                    # against a zero peer partial.
                    # other rank: validates the peer SDMA payload + reduction
                    # against a zero local partial. Bit equality therefore
                    # isolates corruption in either half of the data path.
                    if not torch.equal(output, expected):
                        max_abs = (
                            output.float() - expected.float()
                        ).abs().max().item()
                        role = "local" if rank == source_rank else "peer-payload"
                        raise AssertionError(
                            f"rank={rank} source={source_rank} role={role} "
                            f"q={q} M={m}: not bit-exact, max_abs={max_abs}"
                        )
            result_q.put((rank, True, ""))
        finally:
            try:
                api.shutdown()
            finally:
                api.destroy_stream(stream)
    except BaseException as exc:
        result_q.put((rank, False, repr(exc)))
        raise


def run_q(q: int, shapes: list[int]) -> None:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    result_q = ctx.Queue()
    ps = [
        ctx.Process(target=_run_rank, args=(rank, q, shapes, barrier, result_q))
        for rank in (0, 1)
    ]
    for p in ps:
        p.start()
    results = [result_q.get(timeout=900) for _ in range(2)]
    for p in ps:
        p.join(timeout=30)
        if p.exitcode != 0:
            raise RuntimeError(
                f"rank process pid={p.pid} exitcode={p.exitcode}; results={results}"
            )
    failures = [item for item in results if not item[1]]
    if failures:
        raise RuntimeError(f"q={q} failed: {failures}")
    print(f"PASS isolated local-MM/peer-payload q={q}: {shapes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(1, 2, 4), action="append")
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in ("0,1", "0, 1"):
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES=0,1 for the 310P3 Duo ST")

    selected = args.q or [2, 1, 4]
    matrices = {
        1: [255, 256, 257],
        2: [255, 256, 257, 511, 512, 513],
        4: [255, 256, 257, 1023, 1024, 1025],
    }
    for q in selected:
        # The persistent runtime fixes q for a worker lifetime, so every q is
        # intentionally tested in a fresh pair of processes.
        run_q(q, matrices[q])


if __name__ == "__main__":
    main()
