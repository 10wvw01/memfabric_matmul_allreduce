#include "memfabric_matmul_allreduce_tiling.h"

#include "memfabric_matmul_allreduce_contract.h"

namespace memfabric_mc2 {

int BuildTiling(uint32_t m, uint32_t q, MemFabricMatmulAllReduceTilingData* out)
{
    if (out == nullptr || m == 0 || !IsValidBatchBaseMCount(q)) {
        return -1;
    }
    const uint32_t batch_m = BatchM(q);
    out->m = m;
    out->k = kLocalK;
    out->n = kN;
    out->batch_m = batch_m;
    out->batch_count = (m + batch_m - 1U) / batch_m;
    out->tail_rows = m % batch_m;
    out->block_dim = kCooperativeCores;
    return 0;
}

}  // namespace memfabric_mc2
