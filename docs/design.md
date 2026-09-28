# MemFabric MatMulAllReduce V1 设计

## 1. 目标

把 vllm-ascend 中现有 310P3 `o_proj + TP=2 reduction` 实现迁移为独立的 CANN 9.1 registry-invoke MC² 自定义算子：

```text
local = MatMul(x, weight)
out   = AllReduceSum(local, world_size=2)
```

MemFabric 只作为 AllReduce transport backend。底层不得出现 Qwen / o_proj 语义。

## 2. 交付边界

独立算子工程负责：

- `op_host`：OpDef、shape/dtype 校验、tiling/workspace；
- `op_kernel`：8-core cooperative MM、MemFabric signal/wait/quiet、FP16 reduce；
- MemFabric host lifecycle：context、arena、geometry、credit、Graph warmup；
- ACLNN API；
- `.run` 安装包；
- UT/ST、精度与性能 benchmark。

vllm-ascend 只负责：

- 判断目标层/shape/TP 是否可用；
- 调用已安装的 ACLNN API；
- 已融合时禁止重复 HCCL all-reduce；
- unavailable / small-M 时回退 stock linear + HCCL。

## 3. V1 固定合同

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

## 4. 标准工程结构

```text
memfabric_matmul_allreduce/
├── CMakeLists.txt
├── CMakePresets.json
├── build.sh
├── op_host/
│   ├── memfabric_matmul_allreduce_def.cpp
│   ├── memfabric_matmul_allreduce_tiling.h
│   └── memfabric_matmul_allreduce_tiling.cpp
├── op_kernel/
│   ├── memfabric_matmul_allreduce.cpp
│   ├── matmul_pipeline.h
│   └── memfabric_collective.h
├── runtime/
│   ├── memfabric_runtime.h
│   └── memfabric_runtime.cpp
├── tests/
│   ├── ut/
│   ├── st/
│   └── perf/
└── docs/
```

其中 `runtime/` 是本算子因 MemFabric 有状态通信所需的 host runtime；它属于算子包，不属于 vllm-ascend。

## 5. ACLNN 接口

目标 API：

```text
aclnnMemFabricMatmulAllReduceGetWorkspaceSize(...)
aclnnMemFabricMatmulAllReduce(...)
```

输入至少包含 `x`、`weight`、`tpRank`、`batchBaseMCount`；输出为已完成 TP SUM 的 FP16 tensor。

V1 API 不暴露 signal/wait/arena/credit 等协议细节。

## 6. 数据流

保持当前 v8 路径：

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

## 7. MemFabric 边界

- 只使用 public host API 与 public device `signal/wait/quiet`；
- 不访问 ring/mailbox/SQE/AICPU orchestrator 私有布局；
- data signal 仅一个 core 拥有；
- mail 严格校验 status/dst/len/imm；
- protocol failure 后 context poisoned，不继续复用。

MemFabric headers / `libmf_smem.so` 是算子工程构建依赖。运行期依赖由 custom OPP 安装合同承接，不再由 vllm-ascend CMake 查找或链接。

## 8. Graph / lifecycle

- capture 前完成 context、scratch、protocol、kernel warmup；
- capture 内禁止 create/malloc/host barrier/lazy warmup；
- eager 固定单 execution stream；
- Graph-used context 保持进程生命周期；
- worker teardown 由算子 runtime 负责。

## 9. vllm-ascend 适配

vllm 侧保留一个 thin adapter：

```text
Python routing
  -> torch.ops._C_ascend.memfabric_matmul_allreduce
  -> thin C++ ACLNN adapter
  -> aclnnMemFabricMatmulAllReduce
```

删除 vllm 仓库内：

- MemFabric public header/lib 发现逻辑；
- `.asc` 编译；
- MemFabric runtime / adapter 实现；
- 算子内部 batch/wave/protocol 实现。

Python routing 仍保留模型语义判断及 `MIN_M` fallback。

## 10. 实施 Gate

1. 标准工程可配置/编译/安装；
2. ACLNN 单算子功能可调用；
3. TP=2 精度/协议/稳定性通过；
4. 独立性能压测通过；
5. vllm-ascend thin adapter 接入；
6. Qwen eager / ACL Graph 回归；
7. 删除旧内嵌实现且 feature-off/fallback 正常。
