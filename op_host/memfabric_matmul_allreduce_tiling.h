#pragma once

#include <cstdint>

namespace memfabric_mc2 {

struct MemFabricMatmulAllReduceTilingData {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t batch_m;
    uint32_t batch_count;
    uint32_t tail_rows;
    uint32_t block_dim;
};

}  // namespace memfabric_mc2
