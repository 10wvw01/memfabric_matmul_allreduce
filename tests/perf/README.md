# Standalone performance benchmark contract

Benchmark must run without vLLM/Qwen.

Compare:

- stock FP16 MatMul + HCCL AllReduce
- MemFabricMatmulAllReduce custom op

Required M set: 256, 512, 1024, 2048, 4096, 6144, 8192.
Required q set: 1, 2, 4.

Report warm median/p90/p99, local-MM time, communication+reduce time, end-to-end time, speedup, peak memory and CANN timeline. Run at least 1000 steady iterations for representative shapes. Performance claims must include CANN/Driver/Firmware/MemFabric/operator SHA.
