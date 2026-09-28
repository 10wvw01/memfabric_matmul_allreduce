#include "memfabric_matmul_allreduce_contract.h"
#include "memfabric_matmul_allreduce_tiling.h"

// CANN 9.1 registry-invoke bootstrap.
// The concrete OpDef/ACLNN registration macros are intentionally isolated in
// this translation unit so they can be wired against the target CANN 9.1 SDK
// without coupling runtime/kernel implementation to framework headers.

namespace memfabric_mc2 {

int ValidateContract(uint32_t rank, uint32_t world_size, uint32_t q,
                     uint32_t k, uint32_t n)
{
    if (world_size != kWorldSize || rank >= kWorldSize) {
        return -1;
    }
    if (!IsValidBatchBaseMCount(q)) {
        return -2;
    }
    if (k != kLocalK || n != kN) {
        return -3;
    }
    return 0;
}

}  // namespace memfabric_mc2
