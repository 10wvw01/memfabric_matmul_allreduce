# 方案/代码审视记录

## Round 1 — 所有权与依赖边界

结论：通过，需整改一项。

发现：旧 vllm-ascend CMake 直接发现 MemFabric headers/lib 并编译 device `.asc`，导致框架仓拥有算子实现。

整改：新工程接管 MemFabric build/runtime 依赖和 kernel 编译；vllm-ascend 只保留 ACLNN thin adapter 与 routing。

## Round 2 — CANN registry-invoke 结构与 lifecycle

结论：通过，保留内部 runtime 层。

发现：MemFabric context/arena/geometry/credit/Graph warmup 是持久状态，不能强行压进普通无状态 `op_host`。

整改：标准 `op_host + op_kernel` 之外增加 operator-owned `runtime/`；`op_host` 只负责 OpDef/shape/dtype/tiling/workspace，runtime 不暴露给 vllm。

## Round 3 — API / 正确性 / 性能可验证性

结论：设计可实施，编码 Gate 如下。

1. V1 ACLNN API 只表达 MatMulAllReduce 数学语义和必要 rank/q，不暴露 MemFabric protocol；
2. V1 保持现有 v8 大 M pipeline，不同时引入 small-M decode 算法，避免迁移与优化耦合；
3. 精度测试必须分三层：local MM、peer payload、final SUM；只测 final output 不足以定位通信错；
4. 性能测试必须独立于 vLLM，A/B 为 stock MatMul+HCCL 与 custom op；
5. 迁移性能 Gate：不得劣于当前 v8 同 shape，实测结果绑定软件/固件/operator SHA；
6. tail、multiple-wave、stale generation、mail mismatch、poisoned context 属 P0；
7. ACL Graph capture 前 warmup/lifecycle 必须保持现有 contract。

## 静态代码审视状态

- Design review: 3/3 complete.
- Implementation static review: pending until code skeleton/migration lands.
- Hardware execution: not claimed in this repository review; requires target 310P3 + CANN 9.1.0.
