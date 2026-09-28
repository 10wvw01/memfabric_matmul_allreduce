#include "memfabric_runtime.h"

#include "memfabric310p_adapter_api.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace memfabric_mc2 {
namespace {
constexpr uint32_t kBaseM = 256;
constexpr uint32_t kWidth = 2048;
constexpr uint32_t kMaxArenaRows = 8192;
constexpr uint32_t kEagerGenerationBase = 0x40000000u;
constexpr uint64_t kRowBytes =
    static_cast<uint64_t>(kWidth) * sizeof(uint16_t);
constexpr uint64_t kDefaultLocalPoolBytes = 96ULL * 1024ULL * 1024ULL;
constexpr uint64_t kPoolHeadroomBytes = 1ULL * 1024ULL * 1024ULL;
constexpr const char* kDefaultStoreUrl = "tcp://127.0.0.1:8581";

thread_local std::string g_last_error;

struct RuntimeState {
    std::mutex mutex;
    mfmc2_context_t* ctx = nullptr;
    mfmc2_layout_t layout{};
    int rank = -1;
    uint32_t batch_basem_count = 0;
    uint32_t batch_m = 0;
    bool poisoned = false;
    std::string failure_reason;
    uint64_t producer_scratch = 0;
    bool producer_warmed = false;
    bool waiter_warmed = false;
    bool add_warmed = false;
    bool protocol_warmed = false;
    bool protocol_initialized = false;
    aclrtStream eager_stream = nullptr;
    bool graph_capture_seen = false;
    uint32_t wave_seq = 0;
};

RuntimeState& State()
{
    static RuntimeState state;
    return state;
}

int Fail(int code, const std::string& message)
{
    g_last_error = message;
    return code;
}

std::string WithRet(const char* what, int ret)
{
    return std::string(what) + " failed, ret=" + std::to_string(ret);
}

const char* FirstEnv(const char* preferred, const char* compatibility)
{
    const char* value = std::getenv(preferred);
    if (value != nullptr && *value != '\0') return value;
    value = std::getenv(compatibility);
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

int ParseU64Env(const char* preferred, const char* compatibility,
                uint64_t fallback, uint64_t* out)
{
    const char* value = FirstEnv(preferred, compatibility);
    if (value == nullptr) {
        *out = fallback;
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    if (errno != 0 || end == value || *end != '\0') {
        return Fail(-20, std::string("invalid ") + preferred + "=" + value);
    }
    *out = static_cast<uint64_t>(parsed);
    return 0;
}

int InitContextLocked(RuntimeState& state, int rank, uint32_t q)
{
    if (mfmc2_adapter_abi_version() != MFMC2_ADAPTER_ABI_VERSION) {
        return Fail(-21, "MemFabric adapter ABI mismatch");
    }
    if (rank != 0 && rank != 1) {
        return Fail(-22, "MemFabricMatmulAllReduce only supports TP rank 0/1");
    }
    if (q != 1 && q != 2 && q != 4) {
        return Fail(-23, "batch_basem_count must be one of {1,2,4}");
    }
    const uint32_t batch_m = kBaseM * q;

    if (state.ctx != nullptr) {
        if (state.rank != rank || state.batch_basem_count != q) {
            return Fail(-24,
                        "runtime rank or batch size changed; restart both TP workers");
        }
        return 0;
    }

    uint64_t local_pool_bytes = 0;
    int ret = ParseU64Env("MFMC2_LOCAL_BYTES",
                          "VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES",
                          kDefaultLocalPoolBytes, &local_pool_bytes);
    if (ret != 0) return ret;

    const uint64_t min_bytes =
        kPoolHeadroomBytes + 2 * static_cast<uint64_t>(batch_m) * kRowBytes + 8;
    if (local_pool_bytes <= min_bytes) {
        return Fail(-25, "MemFabric local pool is too small for send/recv arenas");
    }
    const uint64_t rows_by_budget =
        (local_pool_bytes - kPoolHeadroomBytes - 8) / (2 * kRowBytes);
    uint32_t arena_rows = static_cast<uint32_t>(
        std::min<uint64_t>(rows_by_budget, kMaxArenaRows));
    arena_rows = (arena_rows / batch_m) * batch_m;
    if (arena_rows < batch_m) {
        return Fail(-26, "MemFabric arena budget is smaller than one batch");
    }

    const char* store_url = FirstEnv(
        "MFMC2_STORE_URL", "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL");
    if (store_url == nullptr) store_url = kDefaultStoreUrl;

    ret = mfmc2_create(rank, 2, store_url, local_pool_bytes, arena_rows,
                       batch_m, &state.ctx);
    if (ret != 0 || state.ctx == nullptr) {
        state.ctx = nullptr;
        return Fail(-27, WithRet("mfmc2_create", ret));
    }
    ret = mfmc2_get_layout(state.ctx, &state.layout);
    if (ret != 0 || state.layout.pool_base == 0 ||
        state.layout.batch_m != batch_m ||
        state.layout.arena_rows != arena_rows ||
        state.layout.max_batches == 0 ||
        state.layout.max_batches != arena_rows / batch_m ||
        state.layout.batch_bytes != static_cast<uint64_t>(batch_m) * kRowBytes) {
        (void)mfmc2_destroy(state.ctx);
        state.ctx = nullptr;
        return Fail(-28, "MemFabric returned invalid public pool geometry");
    }

    state.rank = rank;
    state.batch_basem_count = q;
    state.batch_m = batch_m;

    static bool atexit_registered = false;
    if (!atexit_registered) {
        if (std::atexit([]() { (void)Shutdown(); }) != 0) {
            return Fail(-29, "failed to register MemFabric runtime shutdown hook");
        }
        atexit_registered = true;
    }
    return 0;
}

int WarmProducerLocked(RuntimeState& state, aclrtStream stream)
{
    if (state.producer_warmed) return 0;
    for (int i = 0; i < 2; ++i) {
        const int ret = mfmc2_warmup_producer_async(state.ctx, stream);
        if (ret != 0) return Fail(-30, WithRet("producer warmup", ret));
    }
    const aclError ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return Fail(-31, WithRet("producer warmup sync", ret));
    state.producer_warmed = true;
    return 0;
}

int WarmWaiterLocked(RuntimeState& state, aclrtStream stream)
{
    if (state.waiter_warmed) return 0;
    for (int i = 0; i < 2; ++i) {
        const int ret = mfmc2_warmup_waiter_async(state.ctx, stream);
        if (ret != 0) return Fail(-32, WithRet("waiter warmup", ret));
    }
    if (!state.add_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int ret = mfmc2_warmup_add_async(state.ctx, stream);
            if (ret != 0) return Fail(-33, WithRet("add warmup", ret));
        }
        state.add_warmed = true;
    }
    if (!state.protocol_warmed) {
        for (int i = 0; i < 2; ++i) {
            const int ret = mfmc2_warmup_protocol_async(state.ctx, stream);
            if (ret != 0) return Fail(-34, WithRet("protocol warmup", ret));
        }
        state.protocol_warmed = true;
    }
    const aclError ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) return Fail(-35, WithRet("wait/add warmup sync", ret));
    state.waiter_warmed = true;
    return 0;
}

int ValidateStreamLocked(RuntimeState& state, aclrtStream stream, bool capturing)
{
    if (stream == nullptr) return Fail(-36, "a valid ACL stream is required");
    if (capturing) {
        state.graph_capture_seen = true;
        return 0;
    }
    if (state.eager_stream == nullptr) {
        state.eager_stream = stream;
        return 0;
    }
    if (state.eager_stream != stream) {
        return Fail(-37,
                    "eager execution is single-stream by contract; restart worker on stream change");
    }
    return 0;
}

int RequireCaptureReadyLocked(const RuntimeState& state)
{
    if (state.ctx == nullptr || !state.protocol_initialized ||
        state.producer_scratch == 0 || !state.producer_warmed ||
        !state.waiter_warmed || !state.add_warmed || !state.protocol_warmed) {
        return Fail(-38,
                    "ACL Graph capture requires prior eager runtime initialization/warmup");
    }
    return 0;
}

int InitializeProtocolLocked(RuntimeState& state, aclrtStream stream)
{
    if (state.protocol_initialized) return 0;
    int ret = WarmWaiterLocked(state, stream);
    if (ret != 0) return ret;

    ret = mfmc2_prepare_wave_async(state.ctx, 1u, stream);
    if (ret != 0) return Fail(-39, WithRet("initial prepare_wave", ret));
    ret = mfmc2_control_barrier(state.ctx);
    if (ret != 0) return Fail(-40, WithRet("control barrier", ret));
    ret = mfmc2_exchange_geometry(state.ctx);
    if (ret != 0) return Fail(-41, WithRet("geometry exchange", ret));
    ret = mfmc2_get_layout(state.ctx, &state.layout);
    if (ret != 0) return Fail(-42, WithRet("geometry refresh", ret));
    ret = mfmc2_init_credit_async(state.ctx, stream);
    if (ret != 0) return Fail(-43, WithRet("initial credit", ret));

    const aclError sync_ret = aclrtSynchronizeStream(stream);
    if (sync_ret != ACL_SUCCESS) {
        return Fail(-44, WithRet("initial protocol sync", sync_ret));
    }
    state.protocol_initialized = true;
    return 0;
}

int MarkPoisoned(RuntimeState& state, int code)
{
    state.poisoned = true;
    if (state.failure_reason.empty()) state.failure_reason = g_last_error;
    return code;
}
}  // namespace

int Execute(const LaunchParams& params, aclrtStream stream)
{
    g_last_error.clear();
    if (params.rows < 0) return Fail(-1, "rows must be non-negative");
    if (params.rows == 0) return 0;
    if (params.x == 0 || params.weight == 0 || params.output == 0) {
        return Fail(-2, "x/weight/output device addresses must be non-zero");
    }
    if (params.rank != 0 && params.rank != 1) return Fail(-3, "rank must be 0 or 1");
    if (params.batch_basem_count != 1 && params.batch_basem_count != 2 &&
        params.batch_basem_count != 4) {
        return Fail(-4, "batch_basem_count must be one of {1,2,4}");
    }

    RuntimeState& state = State();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.poisoned) {
        return Fail(-5, std::string("runtime is poisoned; restart both TP workers: ") +
                            state.failure_reason);
    }
    if (params.graph_capturing && state.ctx == nullptr) {
        return Fail(-6, "MemFabric context must be initialized before graph capture");
    }

    int ret = InitContextLocked(state, params.rank, params.batch_basem_count);
    if (ret != 0) return MarkPoisoned(state, ret);
    ret = ValidateStreamLocked(state, stream, params.graph_capturing);
    if (ret != 0) return MarkPoisoned(state, ret);

    const uint64_t scratch_bytes =
        static_cast<uint64_t>(state.batch_m) * kRowBytes;
    if (state.producer_scratch == 0) {
        if (params.graph_capturing) {
            return MarkPoisoned(
                state, Fail(-7, "tail scratch must be allocated before graph capture"));
        }
        void* scratch = nullptr;
        const aclError alloc_ret =
            aclrtMalloc(&scratch, scratch_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (alloc_ret != ACL_SUCCESS || scratch == nullptr) {
            return MarkPoisoned(state, Fail(-8, WithRet("producer scratch alloc", alloc_ret)));
        }
        state.producer_scratch = reinterpret_cast<uint64_t>(scratch);
    }

    if (params.graph_capturing) {
        ret = RequireCaptureReadyLocked(state);
    } else {
        ret = WarmProducerLocked(state, stream);
        if (ret == 0) ret = InitializeProtocolLocked(state, stream);
    }
    if (ret != 0) return MarkPoisoned(state, ret);

    const int64_t max_rows_per_wave = state.layout.arena_rows;
    for (int64_t wave_start = 0; wave_start < params.rows;
         wave_start += max_rows_per_wave) {
        const int64_t wave_rows =
            std::min<int64_t>(max_rows_per_wave, params.rows - wave_start);
        const uint32_t batches = static_cast<uint32_t>(
            (wave_rows + state.batch_m - 1) / state.batch_m);
        if (batches == 0 || batches > state.layout.max_batches) {
            return MarkPoisoned(state, Fail(-9, "invalid wave batch count"));
        }

        ret = mfmc2_prepare_wave_async(
            state.ctx, params.graph_capturing ? 1u : 0u, stream);
        if (ret != 0) {
            return MarkPoisoned(state, Fail(-10, WithRet("prepare_wave", ret)));
        }
        ret = mfmc2_gate_async(state.ctx, stream);
        if (ret != 0) {
            return MarkPoisoned(state, Fail(-11, WithRet("gate", ret)));
        }

        const uint64_t x_wave =
            params.x + static_cast<uint64_t>(wave_start) * kRowBytes;
        uint32_t wave_generation = 0;
        if (!params.graph_capturing) {
            ++state.wave_seq;
            if (state.wave_seq < kEagerGenerationBase || state.wave_seq == 0u) {
                state.wave_seq = kEagerGenerationBase;
            }
            wave_generation = state.wave_seq;
        }

        auto ValidRows = [&](uint32_t batch) -> uint32_t {
            const int64_t begin = static_cast<int64_t>(batch) * state.batch_m;
            return static_cast<uint32_t>(
                std::min<int64_t>(state.batch_m, wave_rows - begin));
        };

        auto EnqueueProducer = [&](uint32_t batch) -> int {
            const uint32_t valid_rows = ValidRows(batch);
            uint64_t x_batch =
                x_wave + static_cast<uint64_t>(batch) * state.batch_m * kRowBytes;
            if (valid_rows != state.batch_m) {
                aclError acl_ret = aclrtMemsetAsync(
                    reinterpret_cast<void*>(state.producer_scratch), scratch_bytes,
                    0, scratch_bytes, stream);
                if (acl_ret != ACL_SUCCESS) {
                    return Fail(-12, WithRet("tail scratch memset", acl_ret));
                }
                acl_ret = aclrtMemcpyAsync(
                    reinterpret_cast<void*>(state.producer_scratch), scratch_bytes,
                    reinterpret_cast<const void*>(x_batch),
                    static_cast<uint64_t>(valid_rows) * kRowBytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
                if (acl_ret != ACL_SUCCESS) {
                    return Fail(-13, WithRet("tail scratch copy", acl_ret));
                }
                x_batch = state.producer_scratch;
            }
            const int producer_ret = mfmc2_direct_producer_async(
                state.ctx, x_batch, params.weight, batch,
                params.graph_capturing ? (batch + 1) : wave_generation, stream);
            return producer_ret == 0
                       ? 0
                       : Fail(-14, WithRet("direct producer", producer_ret));
        };

        auto EnqueueWaitAdd = [&](uint32_t batch) -> int {
            int batch_ret = mfmc2_wait_batch_async(state.ctx, batch, stream);
            if (batch_ret != 0) return Fail(-15, WithRet("wait batch", batch_ret));
            const uint64_t out_batch =
                params.output +
                static_cast<uint64_t>(
                    wave_start + static_cast<int64_t>(batch) * state.batch_m) *
                    kRowBytes;
            batch_ret = mfmc2_add_batch_async(
                state.ctx, out_batch, batch, ValidRows(batch), stream);
            return batch_ret == 0
                       ? 0
                       : Fail(-16, WithRet("add batch", batch_ret));
        };

        ret = EnqueueProducer(0);
        if (ret != 0) return MarkPoisoned(state, ret);
        for (uint32_t batch = 1; batch < batches; ++batch) {
            ret = EnqueueProducer(batch);
            if (ret != 0) return MarkPoisoned(state, ret);
            ret = EnqueueWaitAdd(batch - 1);
            if (ret != 0) return MarkPoisoned(state, ret);
        }
        ret = EnqueueWaitAdd(batches - 1);
        if (ret != 0) return MarkPoisoned(state, ret);

        ret = mfmc2_quiet_async(state.ctx, stream);
        if (ret != 0) return MarkPoisoned(state, Fail(-17, WithRet("quiet", ret)));
        ret = mfmc2_ack_async(state.ctx, stream);
        if (ret != 0) return MarkPoisoned(state, Fail(-18, WithRet("ack", ret)));
    }
    return 0;
}

int Shutdown()
{
    RuntimeState& state = State();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.ctx == nullptr) return 0;

    if (state.graph_capture_seen) {
        // Graph replay bypasses this host runtime. Without a device-wide proof
        // that all replay streams are quiescent, process-lifetime ownership is
        // safer than destroying the external pool.
        return 0;
    }

    if (state.eager_stream != nullptr) {
        const int quiet_ret = mfmc2_quiet_async(state.ctx, state.eager_stream);
        if (quiet_ret != 0) {
            state.poisoned = true;
            state.failure_reason = "shutdown quiet enqueue failed";
            return Fail(-50, WithRet("shutdown quiet", quiet_ret));
        }
        const aclError sync_ret = aclrtSynchronizeStream(state.eager_stream);
        if (sync_ret != ACL_SUCCESS) {
            state.poisoned = true;
            state.failure_reason = "shutdown stream sync failed";
            return Fail(-51, WithRet("shutdown sync", sync_ret));
        }
    }

    if (state.producer_scratch != 0) {
        (void)aclrtFree(reinterpret_cast<void*>(state.producer_scratch));
        state.producer_scratch = 0;
    }
    const int destroy_ret = mfmc2_destroy(state.ctx);
    state.ctx = nullptr;
    state.layout = {};
    state.rank = -1;
    state.batch_basem_count = 0;
    state.batch_m = 0;
    state.poisoned = false;
    state.failure_reason.clear();
    state.producer_warmed = false;
    state.waiter_warmed = false;
    state.add_warmed = false;
    state.protocol_warmed = false;
    state.protocol_initialized = false;
    state.eager_stream = nullptr;
    state.wave_seq = 0;
    return destroy_ret == 0 ? 0 : Fail(-52, WithRet("mfmc2_destroy", destroy_ret));
}

uint32_t RuntimeAbiVersion()
{
    return kRuntimeAbiVersion;
}

const char* LastError()
{
    return g_last_error.c_str();
}

int DebugSnapshot(uint64_t* words, size_t word_count)
{
    if (words == nullptr || word_count < 24) return Fail(-60, "debug snapshot needs 24 words");
    std::memset(words, 0, word_count * sizeof(uint64_t));

    RuntimeState& state = State();
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.ctx == nullptr) return 0;

    const auto& layout = state.layout;
    auto ArenaWord = [](uint64_t addr, bool* ok) -> uint64_t {
        uint64_t value = 0;
        const aclError ret = aclrtMemcpy(
            &value, sizeof(value), reinterpret_cast<void*>(addr), sizeof(value),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS && ok != nullptr) *ok = false;
        return value;
    };

    bool own_ok = true;
    bool peer_ok = true;
    words[0] = 0x4D464D4332ULL;  // MFMC2
    words[1] = static_cast<uint64_t>(state.rank);
    words[2] = layout.pool_base;
    words[3] = layout.symmetric_size;
    words[4] = layout.local_size;
    words[5] = layout.send_arena;
    words[6] = layout.recv_arena;
    words[7] = layout.peer_recv_arena;
    words[8] = layout.ack_slot;
    words[9] = layout.peer_ack_slot;
    words[10] = layout.arena_bytes;
    words[11] = layout.batch_bytes;
    words[12] = state.protocol_initialized ? 1 : 0;
    words[13] = ArenaWord(layout.send_arena, &own_ok);
    words[14] = ArenaWord(layout.recv_arena, &own_ok);
    words[15] = ArenaWord(layout.peer_recv_arena, &peer_ok);
    words[16] = own_ok ? 1 : 0;
    words[17] = peer_ok ? 1 : 0;
    uint64_t protocol_status = 0;
    const int status_ret = mfmc2_debug_protocol_status(state.ctx, &protocol_status);
    words[18] = protocol_status;
    words[19] = status_ret == 0 ? 1 : 0;
    words[20] = layout.expected_credit_dst;
    words[21] = layout.expected_recv_base;
    words[22] = layout.arena_rows;
    words[23] = layout.batch_m;
    return 0;
}

}  // namespace memfabric_mc2
