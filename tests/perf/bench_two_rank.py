from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "common"))
from mfmc2_ctypes import Mfmc2Api  # noqa: E402

K = N = 2048
NZ_FORMAT = 29


def _make_inputs(rank: int, m: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(3103000 + rank * 10007 + m)
    x = torch.randn((m, K), generator=g, dtype=torch.float32).half().npu()
    w = torch.randn((N, K), generator=g, dtype=torch.float32).mul_(0.02).half().npu()
    return x, torch_npu.npu_format_cast(w, NZ_FORMAT)


def _worker(rank: int, q: int, shapes: list[int], warmup: int, iters: int,
            master_addr: str, master_port: int, result_q: mp.Queue) -> None:
    torch.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=2,
    )
    api = Mfmc2Api()
    stream = api.create_stream()
    try:
        rows = []
        for m in shapes:
            x, w_nz = _make_inputs(rank, m)
            fused_out = torch.empty((m, N), dtype=torch.float16, device=f"npu:{rank}")
            torch.npu.synchronize()

            def stock_once():
                y = torch.nn.functional.linear(x, w_nz)
                dist.all_reduce(y, op=dist.ReduceOp.SUM)
                torch.npu.synchronize()

            def fused_once():
                api.launch(x.data_ptr(), w_nz.data_ptr(), m, rank, q,
                           fused_out.data_ptr(), stream, synchronize=True)

            for _ in range(warmup):
                stock_once()
                fused_once()
            dist.barrier()

            stock_us = []
            fused_us = []
            for _ in range(iters):
                t0 = time.perf_counter_ns()
                stock_once()
                stock_us.append((time.perf_counter_ns() - t0) / 1000.0)

                t0 = time.perf_counter_ns()
                fused_once()
                fused_us.append((time.perf_counter_ns() - t0) / 1000.0)

            rows.append(
                {
                    "m": m,
                    "rank": rank,
                    "stock_us": statistics.median(stock_us),
                    "fused_us": statistics.median(fused_us),
                    "stock_p95_us": sorted(stock_us)[max(0, int(len(stock_us) * 0.95) - 1)],
                    "fused_p95_us": sorted(fused_us)[max(0, int(len(fused_us) * 0.95) - 1)],
                }
            )
        result_q.put((rank, rows, ""))
    except BaseException as exc:
        result_q.put((rank, [], repr(exc)))
        raise
    finally:
        try:
            api.shutdown()
        finally:
            api.destroy_stream(stream)
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--master-port", type=int, default=29610)
    parser.add_argument("--shapes", type=int, nargs="*",
                        default=[512, 1024, 2048, 4096, 6144, 8192])
    args = parser.parse_args()
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in ("0,1", "0, 1"):
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES=0,1")

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    ps = [
        ctx.Process(
            target=_worker,
            args=(r, args.q, args.shapes, args.warmup, args.iters,
                  "127.0.0.1", args.master_port, result_q),
        )
        for r in (0, 1)
    ]
    for p in ps:
        p.start()
    received = [result_q.get(timeout=1800) for _ in range(2)]
    for p in ps:
        p.join(timeout=30)
        if p.exitcode != 0:
            raise RuntimeError(f"benchmark rank exited {p.exitcode}: {received}")
    errors = [x for x in received if x[2]]
    if errors:
        raise RuntimeError(str(errors))

    by_m = {m: [] for m in args.shapes}
    for _, rows, _ in received:
        for row in rows:
            by_m[row["m"]].append(row)

    print("M,stock_us,fused_us,delta_pct,stock_p95_us,fused_p95_us")
    for m in args.shapes:
        # A TP collective completes at the slower rank; report the max rank
        # median/p95 rather than cherry-picking one process.
        stock = max(r["stock_us"] for r in by_m[m])
        fused = max(r["fused_us"] for r in by_m[m])
        stock_p95 = max(r["stock_p95_us"] for r in by_m[m])
        fused_p95 = max(r["fused_p95_us"] for r in by_m[m])
        delta = (fused / stock - 1.0) * 100.0
        print(f"{m},{stock:.3f},{fused:.3f},{delta:+.2f},{stock_p95:.3f},{fused_p95:.3f}")


if __name__ == "__main__":
    main()
