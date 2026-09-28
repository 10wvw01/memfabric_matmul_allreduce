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


def _cpu_pair(rank: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(2026092811 + rank * 100003 + m)
    x = torch.randn((m, K), generator=g, dtype=torch.float32).half()
    w = torch.randn((N, K), generator=g, dtype=torch.float32).mul_(0.02).half()
    return x, w


def _worker(rank: int, q: int, shapes: list[int], iterations: int,
            barrier: mp.Barrier, result_q: mp.Queue) -> None:
    try:
        torch.npu.set_device(rank)
        api = Mfmc2Api()
        stream = api.create_stream()
        cases = {}
        try:
            for m in shapes:
                refs = []
                actual_x = actual_w = None
                for logical_rank in (0, 1):
                    x_cpu, w_cpu = _cpu_pair(logical_rank, m)
                    x_npu = x_cpu.npu()
                    w_nz = torch_npu.npu_format_cast(w_cpu.npu(), NZ_FORMAT)
                    refs.append(torch.nn.functional.linear(x_npu, w_nz))
                    if logical_rank == rank:
                        actual_x, actual_w = x_npu, w_nz
                cases[m] = (
                    actual_x,
                    actual_w,
                    refs[0] + refs[1],
                    torch.empty((m, N), dtype=torch.float16, device=f"npu:{rank}"),
                )
            torch.npu.synchronize()
            barrier.wait()

            for i in range(iterations):
                m = shapes[i % len(shapes)]
                x, w, expected, output = cases[m]
                api.launch(
                    x.data_ptr(), w.data_ptr(), m, rank, q,
                    output.data_ptr(), stream, synchronize=True
                )
                # Check the first/last calls and periodic samples. Every launch
                # is still synchronized, so protocol failures cannot hide in a
                # later iteration.
                if i == 0 or i == iterations - 1 or (i + 1) % 50 == 0:
                    if not torch.equal(output, expected):
                        max_abs = (
                            output.float() - expected.float()
                        ).abs().max().item()
                        raise AssertionError(
                            f"rank={rank} iter={i + 1} q={q} M={m}: "
                            f"not bit-exact, max_abs={max_abs}"
                        )
            barrier.wait()
            result_q.put((rank, True, ""))
        finally:
            try:
                api.shutdown()
            finally:
                api.destroy_stream(stream)
    except BaseException as exc:
        result_q.put((rank, False, repr(exc)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--shapes", type=int, nargs="*", default=[255, 256, 257, 511, 512, 513, 1024]
    )
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") not in ("0,1", "0, 1"):
        raise RuntimeError("set ASCEND_RT_VISIBLE_DEVICES=0,1 for the 310P3 Duo ST")

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    result_q = ctx.Queue()
    ps = [
        ctx.Process(
            target=_worker,
            args=(rank, args.q, args.shapes, args.iterations, barrier, result_q),
        )
        for rank in (0, 1)
    ]
    for p in ps:
        p.start()
    results = [result_q.get(timeout=3600) for _ in range(2)]
    for p in ps:
        p.join(timeout=30)
        if p.exitcode != 0:
            raise RuntimeError(
                f"rank process pid={p.pid} exitcode={p.exitcode}; results={results}"
            )
    failures = [item for item in results if not item[1]]
    if failures:
        raise RuntimeError(f"stability failed: {failures}")
    print(
        f"PASS repeated stability q={args.q} iterations={args.iterations} "
        f"shapes={args.shapes}"
    )


if __name__ == "__main__":
    main()
