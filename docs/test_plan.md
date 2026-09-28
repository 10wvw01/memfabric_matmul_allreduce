# 测试方案

## 1. 功能可用性

- build/package/install：CANN 9.1.0 + 310P3，产出并安装 custom OPP；
- ACLNN symbol/API 可见；
- TP=2 rank0/rank1 均可初始化、调用、销毁；
- q=1/2/4；
- M 边界：1, 255, 256, 257, 511, 512, 513, 1024, 2048, 4096, 6144, 8192；
- tail、multiple batches、multiple waves；
- repeated >=1000；
- mixed-M；
- eager single-stream；
- ACL Graph capture/replay/repeated replay；
- destroy/recreate；
- 错误注入：非法 rank/q/shape/dtype、mail mismatch、credit/protocol failure；失败 context 必须 fail-stop。

## 2. 计算/通信数据准确性

Reference：每 rank stock FP16 matmul，随后 HCCL SUM AllReduce。

每个测试 shape 同时校验：

- fused output vs reference：按当前 FP16 数值合同设置 atol/rtol，并保存 max_abs/max_rel；
- rank0/rank1 最终输出一致；
- 单独 local MM 输出与 stock matmul 对比；
- peer payload 内容与对端 local MM 对应 batch 对比；
- reduce 后每 batch/tail 有效行对比 reference；
- mail `dst/len/imm/status` 全量一致；
- wave credit/generation 不重用陈旧状态；
- tail padding 不污染 valid rows，也不写越界。

必须覆盖随机数据、全零、全一、正负混合、大/小幅值。

## 3. 独立性能压测

不依赖 vLLM/Qwen，提供单算子 benchmark。

A/B：

```text
A = stock FP16 MatMul + HCCL AllReduce
B = MemFabricMatmulAllReduce
```

M：256, 512, 1024, 2048, 4096, 6144, 8192；q=1/2/4。

记录：

- warm median / p50 / p90 / p99 latency；
- local MM latency；
- communication + reduce latency；
- end-to-end op latency；
- speedup vs stock；
- peak memory；
- profiler timeline；
- MM(n+1) || SDMA(n) overlap；
- 1000 次稳定性及最大抖动。

性能 Gate：迁移后同 shape 不得回退于迁移前 v8；正式数字必须绑定 CANN/Driver/Firmware/MemFabric/operator SHA。

## 4. vllm-ascend 回归

- feature/operator unavailable：stock 路径正常；
- eligible full-attention o_proj：调用 ACLNN custom op；
- linear-attention/GDN/其他 Linear：不命中；
- fused output 后不重复 HCCL AllReduce；
- `MIN_M` 以下保持 stock + HCCL；
- Qwen eager correctness；
- ACL Graph correctness/replay；
- 10min repeated requests；
- E2E TTFT/prefill/output throughput 不低于迁移前同 commit 基线。

## 5. 实机执行顺序

```text
package/install smoke
-> ACLNN minimal correctness
-> M=255/256/257
-> q=2 matrix
-> q=1/4
-> repeated 1000
-> mixed-M / fail-stop
-> profiler
-> stock vs fusion benchmark
-> vllm eager
-> ACL Graph
-> 10min stability
```
