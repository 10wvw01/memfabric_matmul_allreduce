# 测试方案

## 1. 功能可用性

- build/package/install：CANN 9.1.0 + 310P3，产出并安装 custom OPP；
- `check_install.py`：`libcust_opapi.so` / bundled `libmf_smem.so` / ELF closure / runtime ABI；
- TP=2 rank0/rank1 均可初始化、调用、销毁；
- q=1/2/4；
- M 边界：255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2048, 4096, 6144, 8192；
- tail、multiple batches、multiple waves；
- repeated >=1000、mixed-M；
- eager single-stream；
- ACL Graph capture/replay/repeated replay；
- 错误参数和 protocol/device fault 后 fail-stop。

对应脚本：

```text
tests/st/check_install.py
tests/st/test_data_path_accuracy.py
tests/st/test_two_rank_accuracy.py
tests/st/test_repeated_stability.py
```

## 2. 计算/通信数据准确性

**最终正确性 oracle 不依赖 HCCL。** 两个 logical rank 的 stock FP16 NZ local MatMul 在每个进程内独立生成，然后做本地 FP16 SUM，作为 fused output reference。

`test_two_rank_accuracy.py` 覆盖完整双 rank 非零数据路径并要求 bit-exact。

`test_data_path_accuracy.py` 用“仅一个 source rank 非零、另一 rank activation 全零”的方式做分层隔离：

- source rank 的 fused output == source local stock MM：验证 local MM / send arena / zero-peer reduce；
- peer rank 的 fused output == source local stock MM：验证 peer SDMA payload / wait / peer reduce；
- source_rank=0/1 双向执行，避免只验证一个 SDMA 方向；
- 覆盖 q=1/2/4 与 batch/tail 临界 M。

这样不需要向生产 runtime 增加测试专用的 arena 导出 ABI，同时能够把“本地计算错误”和“跨 die payload 错误”分别定位。

协议层同时由 device kernel 的 `status/dst/len/imm` 严格校验守护；任一 mismatch 会进入 fail-stop，而不是继续产生输出。

## 3. 稳定性

`test_repeated_stability.py` 默认 q=2、1000 次、mixed-M，所有调用都同步，首尾及每 50 次做 bit-exact 抽检；任何异步 device/protocol fault 都必须在本轮暴露。

实机再补：

- q=1/q=4 各独立进程重复；
- >60s 与 10min；
- peer GVA diverged；
- HCCL 共存场景；
- fault 后双方 worker 重启，不复用 poisoned runtime。

## 4. 独立性能压测

不依赖 vLLM/Qwen：

```text
A = stock FP16 NZ MatMul + HCCL AllReduce
B = MemFabricMatmulAllReduce custom OPP
```

`tests/perf/bench_two_rank.py` 默认覆盖 M=512,1024,2048,4096,6144,8192，支持 q=1/2/4、warmup/iters 参数。TP collective 按较慢 rank 报告 median/p95，避免挑选快 rank。

必须补 CANN profiler：

- end-to-end op latency；
- MM / SDMA / reduce timeline；
- `MM(n+1) || SDMA(n)` overlap；
- fixed overhead；
- peak memory；
- 抖动。

性能 Gate：M>=4096 不得实质劣于 `docs/migration_baseline_v8.md` 中迁移前 v8；正式数字必须绑定 CANN/Driver/Firmware/MemFabric/operator SHA。

## 5. vllm-ascend 回归

- feature off：stock；
- feature on + OPP unavailable/ABI mismatch：仍保持 stock + HCCL，不能提前关闭 `reduce_results`；
- eligible full-attention o_proj：调用 external ACLNN custom op；
- linear-attention/GDN/其他 Linear：不命中；
- fused output 后不重复 HCCL；
- `MIN_M` 以下 stock + HCCL；
- Qwen eager correctness；
- ACL Graph：必须先发生一次真实 fused eager warmup，再 capture/replay；
- 10min repeated requests；
- E2E TTFT/prefill/output throughput 不低于迁移前同 commit 基线。

## 6. 实机执行顺序

```text
build .run
-> install + check_install
-> isolated local-MM/peer-payload accuracy
-> q=2 final correctness matrix
-> q=1/4 final correctness
-> repeated 1000 / mixed-M
-> standalone stock-vs-fused benchmark
-> CANN profiler
-> vllm feature-off / OPP-missing fallback
-> Qwen eager
-> explicit fused eager warmup
-> ACL Graph capture/replay
-> 10min stability
```

静态阶段不得把以上实机步骤标记为 PASS；只有 310P3 目标机输出才能形成最终验证报告。
