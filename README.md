# memfabric_matmul_allreduce

Ascend 310P3 TP=2 `MatMul + MemFabric AllReduce` MC² custom operator.

## V1 contract

- CANN 9.1.0 registry-invoke custom operator
- Hardware: Ascend 310P3 / dav-2002
- TP: 2
- Compute / exchange / reduce: FP16
- V1 geometry: global K=4096, local K=2048, N=2048
- Collective semantic: SUM
- Communication backend: MemFabric public SHM/SDMA APIs only

This repository owns all operator implementation details: op registration, tiling,
Ascend C kernels, MemFabric lifecycle/protocol, ACLNN interface, tests and
benchmarks. `vllm-ascend` only decides when to use the installed operator and
invokes its ACLNN API.

See `docs/design.md`, `docs/test_plan.md`, and `docs/review_report.md`.
