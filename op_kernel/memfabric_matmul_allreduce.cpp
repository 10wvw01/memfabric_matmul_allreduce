#include "memfabric_matmul_allreduce_contract.h"

// Kernel migration target for CANN 9.1 registry-invoke packaging.
// V1 preserves the already-validated 8-core cooperative producer / MemFabric
// wait-reduce / wave-credit algorithm from vllm-ascend. Device implementation
// is migrated in follow-up commits without changing the mathematical contract.

namespace memfabric_mc2 {

struct KernelContract final {
    static constexpr uint32_t kK = kLocalK;
    static constexpr uint32_t kOutputN = kN;
    static constexpr uint32_t kBlockDim = kCooperativeCores;
};

static_assert(KernelContract::kBlockDim == 8, "V1 requires 8 cooperative cores");

}  // namespace memfabric_mc2
