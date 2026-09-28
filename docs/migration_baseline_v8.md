# Pre-migration v8 baseline

This file records the last real-hardware baseline of the embedded vLLM-Ascend implementation before it was migrated into this independent CANN custom OPP. These numbers are **acceptance targets**, not claims that the migrated package has already reproduced them.

## Environment

- Ascend 310P3 Duo, logical device 0/1, TP=2.
- CANN 9.1.0.
- torch 2.10.0 + torch_npu 2.10.0.post4.
- vLLM 0.27.1.
- Qwen3.6-35B-A3B-w8a8, full-attention `o_proj`, local K=2048, N=2048, FP16.
- MemFabric branch: `wgm-dev-310p`.
- Device compile target: `dav-2002`.

## Correctness baseline

The embedded v8 implementation reached bit-exact (`atol=rtol=0`) equality with stock `linear + TP SUM` for:

- q=2: M={255,256,257,511,512,513,1024,2048,4096,6144,8192}.
- q=1 and q=4: nine-shape matrices including both batch boundaries and tails.
- Diverged peer GVA tests at M={511,2048,8192}.
- ACL Graph: M=2048 x32 replay and M=8192 x16 replay, then continued eager execution.
- Source guards and feature-off fallback.

The migrated operator must reproduce these results before it is accepted.

## Performance baseline

Pre-migration head-to-head, q=2, same layer/weights/process, stock = FP16 NZ matmul + HCCL all-reduce:

| M | stock | fused v8 | fused delta |
|---:|---:|---:|---:|
| 512 | 0.481 ms | 0.627 ms | +30.4% |
| 1024 | 0.789 ms | 0.815 ms | +3.2% |
| 2048 | 1.472 ms | 1.471 ms | +0.1% |
| 4096 | 3.284 ms | 2.726 ms | -17.0% |
| 6144 | 5.273 ms | 3.998 ms | -24.2% |
| 8192 | 6.760 ms | 5.258 ms | -22.2% |

Acceptance posture for this migration:

1. No material regression against the v8 fused numbers for M>=4096.
2. Crossover remains approximately M=2048; production routing may conservatively retain `MIN_M=4096`.
3. Profiler must still show the lookahead pipeline: producer(n+1) enqueued before wait/reduce(n), with public MemFabric SDMA overlap.
4. The v8 fixed-overhead fix must remain present: eager ready-cell generation is monotonic from `0x40000000`; the 32 KiB ready-cell memset is graph-only.

## Known target prerequisites

The previous real-hardware stack required MemFabric's 310P AICPU/orchestrator runtime to be correctly installed on the machine. This repository bundles the public `libmf_smem.so` into its custom OPP, but the migration is **not accepted** until `tests/st/check_install.py` and the real-hardware ST confirm that all transitive MemFabric platform/AICPU prerequisites resolve without ad-hoc environment hacks.
