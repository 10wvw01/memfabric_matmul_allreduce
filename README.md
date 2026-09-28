# memfabric_matmul_allreduce

Ascend 310P3 TP=2 `MatMul + MemFabric AllReduce` MC² custom operator.

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

This repository owns the operator data plane, MemFabric lifecycle/protocol, ACLNN facade, CANN packaging, standalone system tests and benchmarks. `vllm-ascend` only owns model routing/fallback and a thin torch-to-ACLNN adapter.

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

The build uses CANN 9.1 `npu_op_package(TYPE RUN)` and emits `build_out/custom_opp_*.run`. It compiles the device pipeline for `dav-2002` and packages `libcust_opapi.so`, `libmfmc2_device.so`, the public ACLNN header, and the public `libmf_smem.so` into one custom OPP.

## Install

```bash
./build_out/custom_opp_*.run --install-path=/opt/memfabric_mc2
source /opt/memfabric_mc2/vendors/memfabric_mc2/bin/set_env.bash
```

The target machine must also satisfy the MemFabric 310P platform/AICPU/orchestrator prerequisites. The custom OPP bundles the public host library, but it does not claim to replace system-level MemFabric device deployment until the real-hardware install check and ST pass.

Validate the installation before any model test:

```bash
python3 tests/st/check_install.py
```

## Standalone correctness

Use the logical Duo devices 0/1:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))

python3 tests/st/test_two_rank_accuracy.py
```

The test calls the installed `libcust_opapi.so` directly; it does not import vLLM. The correctness oracle is the sum of two independently generated stock local FP16 linear results, so HCCL is not part of the oracle. Every case is required to be bit-exact.

## Standalone performance

```bash
python3 tests/perf/bench_two_rank.py --q 2 --warmup 20 --iters 100
```

Baseline is stock NZ `linear + HCCL all_reduce`; fused is the independent custom OPP. Report the slower rank median/p95. Reproduce `docs/migration_baseline_v8.md` before accepting the migration.

## vLLM integration

Install/source this custom OPP **before** starting vLLM. The adapted vLLM branch dynamically verifies both ACLNN symbols and runtime ABI=1. If the package is absent or incompatible, it keeps stock matmul + HCCL and does not disable the generic TP reduction.

Runtime knobs owned by this operator:

```bash
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))
```

The old `VLLM_ASCEND_310P_MEMFABRIC_STORE_URL` and `VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES` names remain accepted by runtime ABI 1 only as compatibility fallbacks; new deployment scripts should use `MFMC2_*`.

## Acceptance sequence

1. build `.run` with CANN 9.1.0;
2. install and run `tests/st/check_install.py`;
3. q=2 full correctness matrix;
4. q=1 and q=4 matrices;
5. repeated >=1000 and mixed-M stability;
6. standalone head-to-head performance and CANN profiler overlap;
7. vLLM feature-off/fallback;
8. Qwen eager;
9. minimal ACL Graph replay, then Qwen ACL Graph;
10. 10-minute stability and performance regression check.

See `docs/design.md`, `docs/test_plan.md`, `docs/review_report.md`, and `docs/migration_baseline_v8.md`.
