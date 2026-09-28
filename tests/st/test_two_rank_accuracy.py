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

K = 2048
N = 2048
NZ_FORMAT = 29


def _cpu_pair(rank: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(20260928 + rank * 100003 + m)
    x = torch.randn((m, K), generator=g, dtype=torch.float32).half()
    w = torch.randn((N, K), generator=g, dtype=torch.float32).mul_(0.02).half()
    return x, w


def _run_rank(rank: int, q: int, shapes: list[int], barrier: mp.Barrier,
              result_q: mp.Queue) -> None:
    try:
        torch.npu.set_device(rank)
        api = Mfmc2Api()
        stream = api.create_stream()
        try:
            for m in shapes:
                # Build exactly the same two-rank reference independently in
                # both processes. This avoids using HCCL in the correctness
                # oracle while still validating local MM + peer payload + SUM.
                refs = []
                actual_x = None
                actual_w_nz = None
                for logical_rank in (0, 1):
                    x_cpu, w_cpu = _cpu_pair(logical_rank, m)
                    x_npu = x_cpu.npu()
                    w_npu = w_cpu.npu()
                    w_nz = torch_npu.npu_format_cast(w_npu, NZ_FORMAT)
                    refs.append(torch.nn.functional.linear(x_npu, w_nz))
                    if logical_rank == rank:
                        actual_x = x_npu
                        actual_w_nz = w_nz
                expected = refs[0] + refs[1]
                output = torch.empty((m, N), dtype=torch.float16, device=f"npu:{rank}")
                torch.npu.synchronize()
                barrier.wait()
                api.launch(
                    actual_x.data_ptr(),
                    actual_w_nz.data_ptr(),
                    m,
                    rank,
                    q,
                    output.data_ptr(),
                    stream,
                )
                barrier.wait()
                if not torch.equal(output, expected):
                    max_abs = (output.float() - expected.float()).abs().max().item()
                    raise AssertionError(
                        f"rank={rank} q={q} M={m}: not bit-exact, max_abs={max_abs}"
                    )
            result_q.put((rank, True, ""))
        finally:
            try:
                api.shutdown()
            finally:
                api.destroy_stream(stream)
    except BaseException as exc:  # propagate child diagnostics to parent
        result_q.put((rank, False, repr(exc)))
        raise


def run_q(q: int, shapes: list[int]) -> None:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    result_q = ctx.Queue()
    ps = [ctx.Process(target=_run_rank, args=(r, q, shapes, barrier, result_q)) for r in (0, 1)]
    for p in ps:
        p.start()
    results = [result_q.get(timeout=900) for _ in range(2)]
    for p in ps:
        p.join(timeout=30)
        if p.exitcode != 0:
            raise RuntimeError(f"rank process pid={p.pid} exitcode={p.exitcode}; results={results}")
    failures = [item for item in results if not item[1]]
    if failures:
        raise RuntimeError(f"q={q} failed: {failures}")
    print(f"PASS q={q}: {shapes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(1, 2, 4), action="append")
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in ("0,1", "0, 1"):
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES=0,1 for the 310P3 Duo ST")

    selected = args.q or [2, 1, 4]
    matrices = {
        2: [1, 255, 256, 257, 511, 512, 513, 1024, 2048, 4096, 6144, 8192],
        1: [1, 255, 256, 257, 511, 512, 513, 1024, 2048, 4096],
        4: [1, 255, 256, 257, 1023, 1024, 1025, 2048, 4096, 8192],
    }
    for q in selected:
        # Changing q requires a fresh process pair by runtime contract.
        run_q(q, matrices[q])


if __name__ == "__main__":
    main()
