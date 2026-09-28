# 方案/代码审视记录

## 方案审视 Round 1 — 所有权与依赖边界

结论：通过，整改完成。

发现：旧 vllm-ascend CMake 直接发现 MemFabric headers/lib 并编译 device `.asc`，导致框架仓拥有算子实现。

整改：独立工程接管 MemFabric build/runtime 依赖、device kernel、arena/protocol/lifecycle；vllm-ascend 只保留 dynamic operator-API thin adapter 与模型 routing/fallback。

## 方案审视 Round 2 — CANN 9.1 接入形态与 lifecycle

结论：通过，设计表述已修正。

发现：迁移前 v8 是有状态 host-orchestrated 多 kernel pipeline。若为了形式上的 registry-invoke 把它改写成自动生成 OpDef + 单 kernel，会同时改变执行模型、Graph lifecycle 和性能基线，不属于等价迁移。

整改：V1 使用 CANN 9.1 custom OPP `.run` + `libcust_opapi.so` 两阶段 operator API façade + operator-owned runtime；保持 v8 数据面不重写。纯生成式 ACLNN/单-kernel registry 作为后续独立优化课题。

重要边界：V1 phase-1 接口传入已经在框架 adapter 中取得的 device address。它采用 ACLNN 同样的 two-phase workspace/executor 调用约定并由 custom OPP loader 发现，但不是 CANN code-gen 生成的 `aclTensor*` 标准 ACLNN。文档和接口注释不得把它描述成“自动生成标准 ACLNN”。这样做是为了不在本次等价迁移中引入 `aclGetRawTensorAddr` phase1/phase2 生命周期风险，也不改写已验证 persistent MemFabric runtime。

## 方案审视 Round 3 — API / 正确性 / 性能可验证性

结论：通过，测试 Gate 已落为可执行脚本。

1. operator API façade 只表达 MatMulAllReduce 数学语义和必要 rank/q，不暴露 MemFabric protocol；
2. V1 保持 v8 大 M pipeline，不混入 small-M decode 算法；
3. 精度测试分 local MM / peer payload / final SUM；
4. correctness oracle 不依赖 HCCL；
5. 性能 A/B 独立于 vLLM，比较 stock NZ MatMul+HCCL 与 custom OPP；
6. tail、multi-wave、generation、strict mail、poisoned context、Graph warmup 均作为 P0 contract；
7. 迁移性能不得实质劣于 pre-migration v8，同次结果必须绑定软件/固件/SHA。

---

## 实现静态审视 Round 1 — 数据面等价迁移

结论：通过；未发现阻塞上机的 P0 静态缺陷。

核对项：

- 8-core cooperative MM 仍为 explicit M partition；
- q=1/2/4 对应 batch_m=256/512/1024，`CONFIG_NORM`；
- core0 仍是唯一 data `signal()` owner；
- 每 core 只 clean 自己的连续 C fragment；
- wait 严格检查 mail `status/dst/len/imm`；
- lookahead=1 顺序仍为 `Producer(n+1)` 在 `Wait/Add(n)` 之前 enqueue；
- wave drain 后才 `quiet`，随后 `ack`；
- eager generation 仍从 `0x40000000` 单调推进；
- tail scratch zero-pad + valid-row reduce 逻辑保持。

整改：原 ST 只能证明 final SUM。新增 `test_data_path_accuracy.py`，用单 source-rank 非零方式双向隔离验证 local MM 与 peer SDMA payload；完整 final SUM 继续由 `test_two_rank_accuracy.py` 验证。

## 实现静态审视 Round 2 — 安装/API/Graph 生命周期

结论：通过；发现并修正文档级 Graph 风险和安装探测兼容性。

发现/整改：

1. 安装检测最初只假设 `ASCEND_CUSTOM_OPP_PATH` 是 vendor root；已同时覆盖 vendor root、direct `op_api/lib` 和默认 OPP vendor 路径。
2. vLLM 的通用 `GetOpApiFuncAddr` 在模块加载时缓存 custom-op path，因此文档明确要求 **source external OPP 后再 import/start vLLM**。
3. 原实机文档默认 `WARMUP_FALLBACK=1`，可能把首次 MemFabric create/warmup 推迟到 Graph capture；已改为 Graph-safe 默认 `0`，并要求 capture 前至少成功一次 fused eager 调用。
4. 新增 installed operator API phase-1 参数校验 smoke，覆盖非法 rank/q/address 和 M=0 no-op contract。
5. CANN build 脚本改为明确的 `binary` → `package` 两阶段顺序。

## 实现静态审视 Round 3 — 测试公平性、稳定性与交付残留

结论：通过；已达到目标机测试前的静态收口条件。

发现/整改：

1. 初版性能脚本 fused 复用预分配 output，而 stock `linear` 每轮分配，A/B 不公平；已改为两侧 timed call 都承担 caller-visible output allocation。
2. 新增 `test_repeated_stability.py`，默认 1000 次 mixed-M，每次同步并周期 bit-exact 抽检。
3. correctness 增补极端 tail `M=1`，并保留 255/256/257、batch boundary、multi-batch、multi-wave。
4. 新增 `tests/run_real_machine_gate.sh`，固定顺序执行 static → install/API → isolated data path → q=2 → q=1/4 → repeated1000 → standalone perf。
5. vllm-ascend 已删除旧 embedded `.asc`、MemFabric host adapter/runtime、专用 CMake；只保留 external-op thin adapter/routing/fallback。
6. 发现 `libmfmc2_device.so` 由 bisheng 直接生成且 `cust_opapi` 曾以绝对文件路径链接，存在 build-machine 绝对路径进入 `DT_NEEDED` 的 P0 风险。已给 device library 加 `SONAME=libmfmc2_device.so`，改为按 library name 链接，并将安装 RPATH 固定为 `$ORIGIN`。
7. `check_install.py` 新增 `readelf -d` gate：`libcust_opapi.so`、`libmfmc2_device.so`、`libmf_smem.so` 任一出现 absolute `DT_NEEDED` 或 `ldd not found` 均直接失败；对应 source guard 已写入 UT。

## 最终静态结论

- Design review：3/3 complete。
- Implementation static review：3/3 complete。
- 独立算子迁移：静态实现完整，具备 build/install/standalone correctness/stability/perf 入口。
- vllm-ascend 适配：静态边界已完成，external OPP unavailable 时不会提前关闭 generic HCCL reduction。
- 当前状态：**READY FOR REAL-HARDWARE VALIDATION**。
- 未声明：CANN 9.1 `.run` 已实际构建成功、310P3 correctness/perf/ACL Graph 已 PASS。这些只能由目标 310P3 环境执行后确认。
