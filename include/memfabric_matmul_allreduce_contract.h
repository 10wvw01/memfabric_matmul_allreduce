#pragma once

#include <cstdint>

namespace memfabric_mc2 {

constexpr uint32_t kWorldSize = 2;
constexpr uint32_t kLocalK = 2048;
constexpr uint32_t kN = 2048;
constexpr uint32_t kBaseM = 256;
constexpr uint32_t kCooperativeCores = 8;
constexpr uint32_t kMaxBatches = 64;
constexpr uint32_t kAdapterAbiVersion = 1;

inline bool IsValidBatchBaseMCount(uint32_t q)
{
    return q == 1 || q == 2 || q == 4;
}

inline uint32_t BatchM(uint32_t q)
{
    return kBaseM * q;
}

}  // namespace memfabric_mc2
