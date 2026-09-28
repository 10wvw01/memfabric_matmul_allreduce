#pragma once

#include <acl/acl_rt.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aclOpExecutor aclOpExecutor;

#if defined(__GNUC__)
#define MFMC2_EXPORT __attribute__((visibility("default")))
#else
#define MFMC2_EXPORT
#endif

// CANN-style two-phase API. V1 intentionally passes device addresses rather
// than aclTensor objects because the operator owns a persistent communication
// runtime and the PyTorch adapter already validates dtype/layout/shape before
// calling this interface. This also avoids querying tensor raw addresses in
// the forbidden interval between ACLNN phase 1 and phase 2.
MFMC2_EXPORT int aclnnMemFabricMatmulAllReduceGetWorkspaceSize(
    uint64_t x_addr,
    uint64_t weight_addr,
    int64_t rows,
    int64_t tp_rank,
    int64_t batch_basem_count,
    uint64_t output_addr,
    bool graph_capturing,
    uint64_t* workspace_size,
    aclOpExecutor** executor);

MFMC2_EXPORT int aclnnMemFabricMatmulAllReduce(
    void* workspace,
    uint64_t workspace_size,
    aclOpExecutor* executor,
    aclrtStream stream);

// Runtime controls are intentionally separate from the data-path ACLNN call.
// vLLM resolves these symbols dynamically from the installed custom OPP.
MFMC2_EXPORT uint32_t mfmc2RuntimeAbiVersion(void);
MFMC2_EXPORT int mfmc2RuntimeShutdown(void);
MFMC2_EXPORT const char* mfmc2RuntimeGetLastError(void);
MFMC2_EXPORT int mfmc2RuntimeDebugSnapshot(uint64_t* words, size_t word_count);

#undef MFMC2_EXPORT

#ifdef __cplusplus
}
#endif
