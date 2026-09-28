#pragma once

#include <acl/acl_rt.h>
#include <cstddef>
#include <cstdint>

namespace memfabric_mc2 {

constexpr uint32_t kRuntimeAbiVersion = 1u;

struct LaunchParams {
    uint64_t x = 0;
    uint64_t weight = 0;
    uint64_t output = 0;
    int64_t rows = 0;
    int32_t rank = -1;
    uint32_t batch_basem_count = 2;
    bool graph_capturing = false;
};

// Enqueue one complete local-MM + TP=2 MemFabric SUM wave sequence on stream.
// Host/runtime initialization is allowed only for eager execution. Graph
// capture requires an already initialized and warmed process runtime.
int Execute(const LaunchParams& params, aclrtStream stream);

// Deterministic teardown for eager-only use. After graph capture has ever been
// observed the MemFabric pool deliberately remains process-lifetime, matching
// the proven v8 safety contract.
int Shutdown();

// Debug/diagnostic helpers exported through libcust_opapi.so. DebugCopyBatch
// is test-only: after the caller has synchronized the execution stream it
// copies one arena slot to host buffers so ST can independently verify the
// local MatMul payload and the peer SDMA payload before reduction.
uint32_t RuntimeAbiVersion();
const char* LastError();
int DebugSnapshot(uint64_t* words, size_t word_count);
int DebugCopyBatch(uint32_t batch_index,
                   uint32_t valid_rows,
                   void* local_host,
                   void* peer_host,
                   size_t bytes);

}  // namespace memfabric_mc2
