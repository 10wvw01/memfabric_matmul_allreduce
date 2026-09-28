# memfabric_matmul_allreduce

Ascend 310P3 TP=2 `MatMul + MemFabric AllReduce` MC² custom operator.

## 使用入口

实机部署、独立算子验证以及 vllm-ascend 接入请直接参考：[`docs/usage.md`](docs/usage.md)。

## V1 contract

- CANN 9.1.0 custom OPP, single `.run` delivery.
- Hardware: Ascend 310P3 / dav-2002.
- TP=2, SUM.
- Compute / exchange / reduce: FP16.
- V1 geometry: global K=4096, local K=2048, N=2048.
- `baseM=256`, `q in {1,2,4}`, default `q=2`.
- Communication backend: MemFabric public SHM/SDMA APIs only.
- 8 AI cores cooperatively execute local MM; core0 is the sole data `signal()` owner.
- Runtime preserves the validated v8 lookahead=1, credit, strict-mail, graph-warmup and fail-stop contracts.

This repository owns the operator data plane, MemFabric lifecycle/protocol, ACLNN façade, CANN packaging, standalone system tests and benchmarks. `vllm-ascend` only owns model routing/fallback and a thin torch-to-ACLNN adapter.

V1 deliberately preserves the proven host-orchestrated multi-kernel pipeline. It uses CANN 9.1 custom OPP packaging plus a two-phase `libcust_opapi.so` ACLNN façade; it does **not** combine the migration with a new single-kernel registry rewrite.

## Build

Build machine prerequisites:

- CANN 9.1.0 toolkit.
- installed `memfabric_hybrid:wgm-dev-310p` public SDK/runtime package.
- `MEMFABRIC_HYBRID_HOME_PATH` exported by that installation.

```bash
source /usr/local/Ascend/cann/set_env.sh
export MEMFABRIC_HYBRID_HOME_PATH=/path/to/installed/memfabric_hybrid
export MAX_JOBS=8
./build.sh
```

`build.sh` follows the CANN RUN-package sequence: configure, `--target binary`, then `--target package`. It emits `build_out/custom_opp_*.run` and packages `libcust_opapi.so`, `libmfmc2_device.so`, the public ACLNN header and public `libmf_smem.so`.

Static/source contracts can be checked without NPU execution:

```bash
python3 -m pytest tests/ut -q
```

## Install

```bash
./build_out/custom_opp_*.run --install-path=/opt/memfabric_mc2
source /opt/memfabric_mc2/vendors/memfabric_mc2/bin/set_env.bash
python3 tests/st/check_install.py
python3 tests/st/test_api_validation.py
```

The target machine must also satisfy the MemFabric 310P platform/AICPU/orchestrator prerequisites. The custom OPP bundles the public host library, but it does not claim to replace system-level MemFabric device deployment. `check_install.py` first verifies ELF closure/API/ABI; functional ST then verifies the actual device path.

## Standalone correctness

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))

# Isolate local MM and peer SDMA payload in both directions.
python3 tests/st/test_data_path_accuracy.py

# Full two-rank nonzero correctness matrix, q=1/2/4.
python3 tests/st/test_two_rank_accuracy.py

# Protocol/arena/generation reuse stability.
python3 tests/st/test_repeated_stability.py --q 2 --iterations 1000
```

Correctness does not use HCCL as the oracle. Each process builds the two stock local FP16 NZ MatMul partials independently and sums them locally. The data-path isolation test makes only one TP rank nonzero at a time: the source rank validates local MM, while the opposite rank validates the exact peer payload. All comparisons are bit-exact, matching the pre-migration v8 contract.

## Standalone performance

```bash
python3 tests/perf/bench_two_rank.py --q 2 --warmup 20 --iters 100
```

Baseline is stock NZ `linear + HCCL all_reduce`; fused is the independent custom OPP. Report the slower rank median/p95. Reproduce `docs/migration_baseline_v8.md` before accepting the migration and capture a CANN profiler timeline proving `MM(n+1) || SDMA(n)` remains present.

## One-command pre-vLLM hardware gate

After build/install/source:

```bash
bash tests/run_real_machine_gate.sh
```

It runs static contracts → installed OPP/API/ABI → isolated local-MM/peer-payload correctness → q=2 → q=1/4 → repeated 1000 → standalone performance. Profiler and vLLM E2E remain explicit subsequent gates.

## vLLM integration

Install/source this custom OPP **before importing/starting vLLM**. The adapted vLLM branch dynamically verifies both ACLNN symbols and runtime ABI=1. If the package is absent or incompatible, it keeps stock matmul + HCCL and does not disable the generic TP reduction.

Operator-owned runtime knobs:

```bash
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))
```

The old `VLLM_ASCEND_310P_MEMFABRIC_STORE_URL` and `VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES` names remain accepted by runtime ABI 1 only as compatibility fallbacks.

## Acceptance sequence

1. build `.run` with CANN 9.1.0;
2. install and run `tests/st/check_install.py` + `tests/st/test_api_validation.py`;
3. isolated local-MM / peer-payload bit-exact test;
4. q=2 full correctness matrix;
5. q=1 and q=4 matrices;
6. repeated >=1000 and mixed-M stability;
7. standalone head-to-head performance and CANN profiler overlap;
8. vLLM feature-off / OPP-missing fallback;
9. Qwen eager;
10. explicit fused eager warmup before any fused ACL Graph capture;
11. ACL Graph replay;
12. 10-minute stability and performance regression check.

See `docs/usage.md`, `docs/design.md`, `docs/test_plan.md`, `docs/review_report.md`, and `docs/migration_baseline_v8.md`.
