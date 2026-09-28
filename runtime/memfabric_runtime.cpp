#include "memfabric_runtime.h"

#include <mutex>

namespace memfabric_mc2 {
namespace {

struct RuntimeState {
    std::mutex mutex;
    bool initialized = false;
    RuntimeConfig config{};
};

RuntimeState& State()
{
    static RuntimeState state;
    return state;
}

bool SameConfig(const RuntimeConfig& a, const RuntimeConfig& b)
{
    return a.rank == b.rank && a.world_size == b.world_size &&
           a.batch_basem_count == b.batch_basem_count &&
           a.local_pool_bytes == b.local_pool_bytes;
}

}  // namespace

int EnsureRuntime(const RuntimeConfig& config, void* acl_stream)
{
    (void)acl_stream;
    if (config.rank < 0 || config.rank >= config.world_size ||
        config.world_size != static_cast<int>(kWorldSize) ||
        !IsValidBatchBaseMCount(config.batch_basem_count) ||
        config.store_url == nullptr) {
        return -1;
    }

    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.initialized) {
        return SameConfig(state.config, config) ? 0 : -2;
    }

    // Bootstrap contract only. The next migration commit moves the existing
    // public MemFabric context/arena/geometry/credit lifecycle here unchanged.
    state.config = config;
    state.initialized = true;
    return 0;
}

int ShutdownRuntime()
{
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.initialized = false;
    state.config = RuntimeConfig{};
    return 0;
}

bool RuntimeAvailable()
{
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.initialized;
}

}  // namespace memfabric_mc2
