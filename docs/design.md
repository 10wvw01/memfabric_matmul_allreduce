# MemFabric MatMulAllReduce V1 设计

## 1. 目标

把 vllm-ascend 中已实机验证的 310P3 `o_proj + TP=2 reduction` 数据面迁移为独立 CANN 9.1 custom OPP：

```text
local = MatMul(x, weight)
out   = AllReduceSum(local, world_size=2)
```

MemFabric 只作为 AllReduce transport backend。独立工程底层不得出现 Qwen / o_proj 语义。

## 2. CANN 9.1 接入形态

V1 使用 CANN 9.1 官方 OPP packaging/build extension：

```text
custom_opp_*.run
  -> vendors/memfabric_mc2/op_api/include/aclnn_*.h
  -> vendors/memfabric_mc2/op_api/lib/libcust_opapi.so
  -> libmfmc2_device.so
  -> bundled public libmf_smem.so
```

对调用方暴露标准两阶段调用形态：

```text
aclnnMemFabricMatmulAllReduceGetWorkspaceSize(...)
aclnnMemFabricMatmulAllReduce(...)
```

**V1 有意不做自动生成 OpDef 的单-kernel registry-invoke 重写。** 原因是迁移前 v8 已验证算法是一个有状态的 host-orchestrated 多 kernel pipeline：一个 ACLNN phase-2 内按顺序 enqueue gate / producer / wait / add / quiet / ack，并维护跨调用的 MemFabric context/arena/credit/Graph warmup。把它强制改成自动生成的单 kernel registry 路径会同时改变执行模型和迁移边界，无法做等价迁移验收。

因此 V1 的原则是：**CANN 9.1 单 `.run` 交付和 ACLNN 可见性标准化，数据面执行模型不重写。** 后续若要转为纯 registry single-kernel，应作为独立优化版本重新验证。

## 3. 交付边界

独立算子工程负责：

- `op_api/`：两阶段 ACLNN façade；
- `runtime/`：MemFabric context、arena、geometry、credit、Graph warmup、fail-stop 和多 kernel 调度；
- `op_kernel/`：8-core cooperative MM、MemFabric signal/wait/quiet、FP16 reduce；
- CANN 9.1 `.run` packaging；
- public MemFabric SDK/runtime build contract；
- 独立 UT/ST、数据路径精度与性能 benchmark。

vllm-ascend 只负责：

- 判断目标层/shape/TP 是否可用；
- 通过 CANN custom OPP loader 动态调用 ACLNN API；
- 已融合时禁止重复 HCCL all-reduce；
- unavailable / small-M 时回退 stock linear + HCCL。

## 4. V1 固定合同

```text
CANN          9.1.0
SoC           Ascend 310P3 / dav-2002
world_size    2
reduce op     SUM
dtype         FP16
local K       2048
N             2048
baseM         256
q             1 / 2 / 4, default 2
batch_m       256 * q
blockDim      8
```

V1 保持当前已验证的大 M 算法，不混入 Decode small-M 新设计。

## 5. 工程结构

```text
memfabric_matmul_allreduce/
├── CMakeLists.txt
├── CMakePresets.json
├── build.sh
├── include/
│   └── aclnn_mem_fabric_matmul_all_reduce.h
├── op_api/
│   └── aclnn_mem_fabric_matmul_all_reduce.cpp
├── op_kernel/
│   └── memfabric310p_device.asc
├── runtime/
│   ├── memfabric_runtime.{h,cpp}
│   └── memfabric310p_adapter{_api.h,.cpp}
├── tests/
│   ├── ut/
│   ├── st/
│   └── perf/
└── docs/
```

`runtime/` 属于算子包，不属于 vllm-ascend。

## 6. ACLNN façade

V1 phase-1 接收已经过框架侧严格验证的 device address、M、rank、q、output address 及 graph-capture 标志，生成本次调用 executor；phase-2 在当前 ACL stream 上调用 operator-owned runtime。

API 不暴露 signal/wait/arena/credit 等 MemFabric 协议细节。runtime ABI 单独版本化，vllm 在关闭 generic TP reduction 前先检查 API symbol + ABI。

## 7. 数据流

保持迁移前 v8 路径：

```text
wave gate
P0: 8-core cooperative MM -> ready -> core0 signal
P1: MM -> signal              || SDMA0
W0 -> A0                      || SDMA1
P2 -> ...
drain wait/add
quiet once per wave
credit ack
```

每个 producer launch 处理一个 `batch_m`；tail zero-pad 到完整 batch，最后只写 valid rows。

## 8. MemFabric 边界

- 只使用 public host API 与 public device `signal/wait/quiet`；
- 不访问 ring/mailbox/SQE/AICPU orchestrator 私有布局；
- data signal 仅一个 core 拥有；
- mail 严格校验 status/dst/len/imm；
- protocol failure 后 context poisoned，不继续复用。

MemFabric headers / `libmf_smem.so` 是算子工程构建依赖。`libmf_smem.so` 随 OPP 放入 `op_api/lib` 并通过 `$ORIGIN` RPATH 解析；其系统级 310P AICPU/orchestrator 前置仍由目标机 MemFabric 部署保证并由安装/ST 检查验证。

## 9. Graph / lifecycle

- capture 前完成 context、scratch、protocol、kernel warmup；
- capture 内禁止 create/malloc/host barrier/lazy warmup；
- eager 固定单 execution stream；
- Graph-used context 保持进程生命周期；
- worker teardown 由算子 runtime/atexit 负责；
- ACL Graph 场景不得启用“跳过所有融合 warmup、首次调用直接发生在 capture 内”的配置。

## 10. vllm-ascend 适配

```text
Python routing
  -> torch.ops._C_ascend.memfabric_matmul_allreduce
  -> thin C++ dynamic ACLNN adapter
  -> installed libcust_opapi.so
  -> operator-owned runtime/data plane
```

vllm 仓库删除：MemFabric public header/lib 发现、`.asc` 编译、MemFabric runtime/adapter、batch/wave/protocol 实现。Python 仅保留模型语义判断、q、`MIN_M` 与安全 fallback。

## 11. 实施 Gate

1. CANN 9.1 `.run` 可配置/编译/安装；
2. ACLNN symbol + runtime ABI 可见；
3. local MM / peer payload / final SUM 分层 bit-exact；
4. q=1/2/4、tail、multi-wave、1000 次稳定性通过；
5. 独立性能压测不低于迁移前 v8；
6. vllm-ascend thin adapter/fallback 通过；
7. Qwen eager / ACL Graph / 10min 实机回归通过。
