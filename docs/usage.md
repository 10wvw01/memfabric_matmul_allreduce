# 使用说明

本文面向 310P3 实机部署与验证，只描述如何构建、安装、验证并接入 vllm-ascend。

## 1. 环境前提

- CANN 9.1.0
- Ascend 310P3，TP=2
- 已部署 `memfabric_hybrid:wgm-dev-310p` 所需系统级运行环境（包含 310P AICPU/orchestrator 前置）
- `vllm-ascend` 使用分支：`feat/310p-memfabric-public-api`
- Qwen3.6 模型路径示例：`/home/models/Qwen3.6-35B-A3B-w8a8`

## 2. 构建独立算子

```bash
cd /path/to/memfabric_matmul_allreduce

source /usr/local/Ascend/cann-9.1.0/set_env.sh
export MEMFABRIC_HYBRID_HOME_PATH=/path/to/installed/memfabric_hybrid
export MAX_JOBS=8

./build.sh
```

构建成功后应生成：

```text
build_out/custom_opp_*.run
```

## 3. 安装算子

```bash
./build_out/custom_opp_*.run --install-path=/opt/memfabric_mc2
source /opt/memfabric_mc2/vendors/memfabric_mc2/bin/set_env.bash
```

安装后先检查 ELF、API 和 runtime ABI：

```bash
python3 tests/st/check_install.py
python3 tests/st/test_api_validation.py
```

注意：`.run` 已包含算子侧需要的 public `libmf_smem.so`，因此 vLLM 启动时不需要单独 source MemFabric SDK 环境；但系统级 MemFabric 310P AICPU/orchestrator 仍必须提前部署正确。

## 4. 独立算子实机验证

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))
```

推荐直接执行完整 gate：

```bash
bash tests/run_real_machine_gate.sh
```

执行顺序为：

```text
static contracts
-> install/API/ABI
-> local MM / peer payload accuracy
-> q=2 correctness
-> q=1/q=4 correctness
-> repeated 1000
-> standalone performance
```

如需逐项执行：

```bash
python3 tests/st/test_data_path_accuracy.py
python3 tests/st/test_two_rank_accuracy.py
python3 tests/st/test_repeated_stability.py --q 2 --iterations 1000
python3 tests/perf/bench_two_rank.py --q 2 --warmup 20 --iters 100
```

正确性测试以 stock local MatMul partial + local SUM 为 oracle，不依赖 HCCL；HCCL 只用于性能 baseline。

## 5. 构建 vllm-ascend

先确保本终端已经 source 独立算子的 `set_env.bash`，再构建/启动 vLLM。

```bash
cd /path/to/vllm-ascend
git checkout feat/310p-memfabric-public-api
git pull

export SOC_VERSION=ascend310p3
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export MAX_JOBS=8

python3 -m pip install -v -e . --no-build-isolation --no-deps
```

vllm-ascend 不再编译或链接 MemFabric kernel/runtime，因此构建 vllm-ascend 时不需要 `MEMFABRIC_HYBRID_HOME_PATH`。

## 6. vLLM Runtime 配置

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

# 独立算子 runtime
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))

# vllm-ascend 路由策略
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT=2
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M=4096
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK=0
```

`q=2` 对应：

```text
batch_m = 512
N = 2048
FP16 payload = 2 MiB
```

合法 q 仅为 `1/2/4`。

## 7. 检查 vLLM 是否能看到独立算子

必须在 import/start vLLM 之前 source：

```bash
source /opt/memfabric_mc2/vendors/memfabric_mc2/bin/set_env.bash
```

然后检查：

```bash
python3 - <<'PY'
import torch
import vllm_ascend.vllm_ascend_C  # noqa

print(
    "external OPP available:",
    torch.ops._C_ascend.memfabric_matmul_allreduce_available(),
)
PY
```

预期：

```text
external OPP available: True
```

如果为 False，不允许目标层关闭 generic TP reduction；vllm-ascend 应自动保持 stock MatMul + HCCL fallback。

## 8. Qwen eager 验证

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=/home/models/Qwen3.6-35B-A3B-w8a8

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b-w8a8 \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --dtype float16 \
  --trust-remote-code \
  --enforce-eager
```

检查：

- 仅 full-attention `self_attn.o_proj` 命中；
- `M < VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M` 时走 stock MatMul + HCCL；
- 命中融合算子后不再执行重复的 generic HCCL all-reduce；
- feature-off 或 external OPP unavailable 时可安全 fallback。

进入 ACL Graph 前，至少完成一次真实 fused eager 调用，使 MemFabric context、scratch、protocol 和 kernel warmup 完成。

## 9. ACL Graph 与最终验收

Eager 通过后去掉 `--enforce-eager`，保持：

```bash
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK=0
```

最终建议按以下顺序验收：

```text
custom OPP build/install
-> standalone correctness
-> q=1/2/4
-> repeated 1000
-> standalone performance
-> CANN profiler
-> OPP-missing fallback
-> Qwen eager
-> fused eager warmup
-> ACL Graph
-> 10min stability
```

Profiler 需要重新确认：

```text
MM(n+1) || SDMA(n)
```

并确认迁移后未重新引入每 wave 的固定清零/同步开销。

只有上述项目全部通过，并记录 CANN/Driver/Firmware、`memfabric_hybrid` SHA、`memfabric_matmul_allreduce` SHA 和 `vllm-ascend` SHA，才视为本次独立算子迁移实机验收完成。
