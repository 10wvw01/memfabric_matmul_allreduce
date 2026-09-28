#include "aclnn_mem_fabric_matmul_all_reduce.h"

#include "memfabric_runtime.h"

#include <new>

namespace {
struct Mfmc2Executor {
    memfabric_mc2::LaunchParams params;
};

Mfmc2Executor* AsExecutor(aclOpExecutor* executor)
{
    return reinterpret_cast<Mfmc2Executor*>(executor);
}
}  // namespace

extern "C" int aclnnMemFabricMatmulAllReduceGetWorkspaceSize(
    uint64_t x_addr,
    uint64_t weight_addr,
    int64_t rows,
    int64_t tp_rank,
    int64_t batch_basem_count,
    uint64_t output_addr,
    bool graph_capturing,
    uint64_t* workspace_size,
    aclOpExecutor** executor)
{
    if (workspace_size == nullptr || executor == nullptr) return -1;
    *workspace_size = 0;
    *executor = nullptr;

    if (rows < 0 || tp_rank < 0 || tp_rank > 1 ||
        (batch_basem_count != 1 && batch_basem_count != 2 &&
         batch_basem_count != 4)) {
        return -2;
    }
    if (rows > 0 && (x_addr == 0 || weight_addr == 0 || output_addr == 0)) {
        return -3;
    }

    auto* impl = new (std::nothrow) Mfmc2Executor();
    if (impl == nullptr) return -4;
    impl->params.x = x_addr;
    impl->params.weight = weight_addr;
    impl->params.output = output_addr;
    impl->params.rows = rows;
    impl->params.rank = static_cast<int32_t>(tp_rank);
    impl->params.batch_basem_count =
        static_cast<uint32_t>(batch_basem_count);
    impl->params.graph_capturing = graph_capturing;

    *executor = reinterpret_cast<aclOpExecutor*>(impl);
    return 0;
}

extern "C" int aclnnMemFabricMatmulAllReduce(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    (void)workspace;
    if (workspace_size != 0 || executor == nullptr || stream == nullptr) {
        delete AsExecutor(executor);
        return -1;
    }

    Mfmc2Executor* impl = AsExecutor(executor);
    const int ret = memfabric_mc2::Execute(impl->params, stream);
    delete impl;
    return ret;
}

extern "C" uint32_t mfmc2RuntimeAbiVersion(void)
{
    return memfabric_mc2::RuntimeAbiVersion();
}

extern "C" int mfmc2RuntimeShutdown(void)
{
    return memfabric_mc2::Shutdown();
}

extern "C" const char* mfmc2RuntimeGetLastError(void)
{
    return memfabric_mc2::LastError();
}

extern "C" int mfmc2RuntimeDebugSnapshot(uint64_t* words, size_t word_count)
{
    return memfabric_mc2::DebugSnapshot(words, word_count);
}
