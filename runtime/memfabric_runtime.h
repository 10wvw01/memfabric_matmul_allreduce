#pragma once

#include <cstdint>

#include "memfabric_matmul_allreduce_contract.h"

namespace memfabric_mc2 {

struct RuntimeConfig {
    int rank = -1;
    int world_size = static_cast<int>(kWorldSize);
    uint32_t batch_basem_count = 2;
    uint64_t local_pool_bytes = 96ULL * 1024ULL * 1024ULL;
    const char* store_url = "tcp://127.0.0.1:8581";
};

// Operator-owned lifecycle. The implementation is process-persistent by V1
// contract and must be capture-ready before ACL Graph capture starts.
int EnsureRuntime(const RuntimeConfig& config, void* acl_stream);
int ShutdownRuntime();
bool RuntimeAvailable();

}  // namespace memfabric_mc2
