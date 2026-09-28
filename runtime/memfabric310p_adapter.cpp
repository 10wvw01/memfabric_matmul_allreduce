#include "memfabric310p_adapter_api.h"

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <smem.h>
#include <smem_shm.h>

#include <cstdint>
#include <new>
#include <vector>

namespace {
constexpr uint64_t kAckSlotBytes = 8ULL;
constexpr uint32_t kOutputWidthElems = 2048;
constexpr uint64_t kOutputRowBytes =
    static_cast<uint64_t>(kOutputWidthElems) * sizeof(uint16_t);
constexpr uint64_t kProducerReadyStrideBytes = 64ULL;
constexpr uint64_t kProducerControlBytes =
    static_cast<uint64_t>(MFMC2_MAX_BATCHES) * MFMC2_COOPERATIVE_CORES *
    kProducerReadyStrideBytes;
constexpr uint64_t kProtocolStatusBytes = 64ULL;

extern "C" int mfmc2_device_launch_direct_producer_async(
    uint64_t pool_base, uint64_t x, uint64_t w, uint64_t send_arena,
    uint64_t peer_recv_arena, uint32_t batch_m, uint32_t batch_index,
    uint32_t generation, uint64_t batch_bytes, uint64_t producer_control,
    uint64_t protocol_status, aclrtStream stream);
extern "C" int mfmc2_device_wait_batch_async(
    uint64_t pool_base, uint64_t expected_recv_base, uint32_t batch_index,
    uint64_t batch_bytes, uint64_t protocol_status, aclrtStream stream);
extern "C" int mfmc2_device_warmup_producer_async(
    uint64_t pool_base, uint32_t batch_m, uint64_t batch_bytes,
    aclrtStream stream);
extern "C" int mfmc2_device_warmup_waiter_async(uint64_t pool_base,
                                                  aclrtStream stream);
extern "C" int mfmc2_device_add_async(uint64_t out, uint64_t send_arena,
                                        uint64_t recv_arena, uint64_t elems,
                                        uint32_t block_count,
                                        aclrtStream stream);
extern "C" int mfmc2_device_warmup_add_async(uint64_t arena,
                                               aclrtStream stream);
extern "C" int mfmc2_device_ack_async(
    uint64_t pool_base, uint64_t ack_slot, uint64_t peer_ack_slot,
    uint64_t protocol_status, uint32_t enable, aclrtStream stream);
extern "C" int mfmc2_device_gate_async(
    uint64_t pool_base, uint64_t expected_credit_dst,
    uint64_t protocol_status, uint32_t enable, aclrtStream stream);
extern "C" int mfmc2_device_quiet_async(
    uint64_t pool_base, uint64_t protocol_status, uint32_t enable,
    aclrtStream stream);

bool IsValidBatchM(uint32_t batch_m)
{
    return batch_m == 256 || batch_m == 512 || batch_m == 1024;
}

void DestroyPartial(mfmc2_context_t* opaque);
}  // namespace

struct mfmc2_context {
    int rank = -1;
    int world_size = 0;
    uint64_t local_size = 0;
    bool smem_inited = false;
    bool shm_inited = false;
    smem_shm_t shm = nullptr;
    void* producer_control = nullptr;
    void* protocol_status = nullptr;
    int device_id = -1;
    mfmc2_layout_t layout{};
};

namespace {
void DestroyPartial(mfmc2_context_t* opaque)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr) return;
    if (ctx->protocol_status != nullptr) {
        (void)aclrtFree(ctx->protocol_status);
        ctx->protocol_status = nullptr;
    }
    if (ctx->producer_control != nullptr) {
        (void)aclrtFree(ctx->producer_control);
        ctx->producer_control = nullptr;
    }
    if (ctx->shm != nullptr) {
        (void)smem_shm_destroy(ctx->shm, 0);
        ctx->shm = nullptr;
    }
    if (ctx->shm_inited) {
        smem_shm_uninit(0);
        ctx->shm_inited = false;
    }
    if (ctx->smem_inited) {
        smem_uninit();
        ctx->smem_inited = false;
    }
    delete ctx;
}
}  // namespace

extern "C" uint32_t mfmc2_adapter_abi_version(void)
{
    return MFMC2_ADAPTER_ABI_VERSION;
}

extern "C" int mfmc2_create(
    int rank, int world_size, const char* store_url, uint64_t local_size,
    uint32_t arena_rows, uint32_t batch_m, mfmc2_context_t** out_ctx)
{
    if (out_ctx == nullptr || store_url == nullptr || world_size != 2 ||
        rank < 0 || rank >= world_size || local_size == 0 ||
        !IsValidBatchM(batch_m) || arena_rows < batch_m ||
        (arena_rows % batch_m) != 0) {
        return -1;
    }
    *out_ctx = nullptr;

    const uint32_t max_batches = arena_rows / batch_m;
    if (max_batches == 0 || max_batches > MFMC2_MAX_BATCHES) return -1;

    const uint64_t arena_bytes =
        static_cast<uint64_t>(arena_rows) * kOutputRowBytes;
    const uint64_t batch_bytes =
        static_cast<uint64_t>(batch_m) * kOutputRowBytes;
    const uint64_t app_bytes = 2 * arena_bytes + kAckSlotBytes;
    if (app_bytes >= local_size) return -2;

    auto* ctx = new (std::nothrow) mfmc2_context();
    if (ctx == nullptr) return -3;
    ctx->rank = rank;
    ctx->world_size = world_size;
    ctx->local_size = local_size;

    int ret = smem_init(0);
    if (ret != 0) {
        delete ctx;
        return ret;
    }
    ctx->smem_inited = true;

    int32_t device_id = -1;
    const aclError device_ret = aclrtGetDevice(&device_id);
    if (device_ret != ACL_SUCCESS || device_id < 0 ||
        device_id > static_cast<int32_t>(UINT16_MAX)) {
        DestroyPartial(ctx);
        return device_ret == ACL_SUCCESS ? -4 : static_cast<int>(device_ret);
    }
    ctx->device_id = device_id;

    smem_shm_config_t cfg;
    ret = smem_shm_config_init(&cfg);
    if (ret != 0) {
        DestroyPartial(ctx);
        return ret;
    }
    cfg.startConfigStoreServer = (rank == 0);

    ret = smem_shm_init(store_url, static_cast<uint32_t>(world_size),
                        static_cast<uint32_t>(rank),
                        static_cast<uint16_t>(device_id), &cfg);
    if (ret != 0) {
        DestroyPartial(ctx);
        return ret;
    }
    ctx->shm_inited = true;

    void* gva = nullptr;
    ctx->shm = smem_shm_create(0, static_cast<uint32_t>(world_size),
                               static_cast<uint32_t>(rank), local_size,
                               SMEMS_DATA_OP_SDMA, 0, &gva);
    if (ctx->shm == nullptr || gva == nullptr) {
        DestroyPartial(ctx);
        return -7;
    }
    if (smem_shm_sdma_get_workspace(ctx->shm) == nullptr) {
        DestroyPartial(ctx);
        return -6;
    }

    const uint64_t symmetric_size = smem_shm_get_symmetric_size(ctx->shm);
    if (symmetric_size == 0) {
        DestroyPartial(ctx);
        return -5;
    }
    if (app_bytes > symmetric_size) {
        DestroyPartial(ctx);
        return -8;
    }

    const uint64_t pool_base = reinterpret_cast<uint64_t>(gva);
    const uint64_t own_segment =
        pool_base + symmetric_size * static_cast<uint64_t>(rank);
    const uint64_t peer_segment =
        pool_base + symmetric_size * static_cast<uint64_t>(1 - rank);

    ctx->layout.pool_base = pool_base;
    ctx->layout.own_segment = own_segment;
    ctx->layout.peer_segment = peer_segment;
    ctx->layout.symmetric_size = symmetric_size;
    ctx->layout.local_size = local_size;
    ctx->layout.send_arena = own_segment;
    ctx->layout.recv_arena = own_segment + arena_bytes;
    ctx->layout.peer_recv_arena = peer_segment + arena_bytes;
    ctx->layout.ack_slot = own_segment + 2 * arena_bytes;
    ctx->layout.peer_ack_slot = peer_segment + 2 * arena_bytes;
    ctx->layout.arena_bytes = arena_bytes;
    ctx->layout.batch_bytes = batch_bytes;
    ctx->layout.arena_rows = arena_rows;
    ctx->layout.batch_m = batch_m;
    ctx->layout.max_batches = max_batches;

    ret = aclrtMalloc(&ctx->producer_control, kProducerControlBytes,
                      ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS || ctx->producer_control == nullptr) {
        DestroyPartial(ctx);
        return -9;
    }
    ret = aclrtMalloc(&ctx->protocol_status, kProtocolStatusBytes,
                      ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS || ctx->protocol_status == nullptr) {
        DestroyPartial(ctx);
        return -10;
    }

    *out_ctx = ctx;
    return 0;
}

extern "C" int mfmc2_destroy(mfmc2_context_t* opaque)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr) return 0;
    int first_ret = 0;
    if (ctx->protocol_status != nullptr) {
        const aclError ret = aclrtFree(ctx->protocol_status);
        if (ret != ACL_SUCCESS && first_ret == 0) first_ret = ret;
        ctx->protocol_status = nullptr;
    }
    if (ctx->producer_control != nullptr) {
        const aclError ret = aclrtFree(ctx->producer_control);
        if (ret != ACL_SUCCESS && first_ret == 0) first_ret = ret;
        ctx->producer_control = nullptr;
    }
    if (ctx->shm != nullptr) {
        const int ret = smem_shm_destroy(ctx->shm, 0);
        if (ret != 0 && first_ret == 0) first_ret = ret;
        ctx->shm = nullptr;
    }
    if (ctx->shm_inited) {
        smem_shm_uninit(0);
        ctx->shm_inited = false;
    }
    if (ctx->smem_inited) {
        smem_uninit();
        ctx->smem_inited = false;
    }
    delete ctx;
    return first_ret;
}

extern "C" int mfmc2_get_layout(mfmc2_context_t* opaque,
                                  mfmc2_layout_t* out_layout)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || out_layout == nullptr) return -1;
    *out_layout = ctx->layout;
    return 0;
}

extern "C" int mfmc2_debug_protocol_status(mfmc2_context_t* opaque,
                                             uint64_t* out_status)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        out_status == nullptr) {
        return -1;
    }
    const aclError ret = aclrtMemcpy(out_status, sizeof(*out_status),
                                     ctx->protocol_status, sizeof(*out_status),
                                     ACL_MEMCPY_DEVICE_TO_HOST);
    return ret == ACL_SUCCESS ? 0 : static_cast<int>(ret);
}

extern "C" int mfmc2_control_barrier(mfmc2_context_t* opaque)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->shm == nullptr) return -1;
    return smem_shm_control_barrier(ctx->shm);
}

extern "C" int mfmc2_exchange_geometry(mfmc2_context_t* opaque)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->shm == nullptr || ctx->world_size != 2) return -1;

    struct Geometry {
        uint64_t pool_base;
        uint64_t symmetric_size;
    };
    const Geometry mine{ctx->layout.pool_base, ctx->layout.symmetric_size};
    std::vector<char> recv(sizeof(Geometry) * ctx->world_size);
    const int32_t ret = smem_shm_control_allgather(
        ctx->shm, reinterpret_cast<const char*>(&mine), sizeof(mine),
        recv.data(), static_cast<uint32_t>(recv.size()));
    if (ret != 0) return ret;

    const auto* peers = reinterpret_cast<const Geometry*>(recv.data());
    if (peers[ctx->rank].pool_base != mine.pool_base ||
        peers[ctx->rank].symmetric_size != mine.symmetric_size) {
        return -2;
    }
    const uint32_t peer_rank = 1 - static_cast<uint32_t>(ctx->rank);
    if (peers[peer_rank].symmetric_size != mine.symmetric_size) return -3;

    const uint64_t my_segment_in_peer_space =
        peers[peer_rank].pool_base +
        mine.symmetric_size * static_cast<uint64_t>(ctx->rank);
    ctx->layout.expected_credit_dst =
        my_segment_in_peer_space + 2 * ctx->layout.arena_bytes;
    ctx->layout.expected_recv_base =
        my_segment_in_peer_space + ctx->layout.arena_bytes;
    return 0;
}

extern "C" int mfmc2_prepare_wave_async(mfmc2_context_t* opaque,
                                          uint32_t clear_wave_control,
                                          void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr ||
        ctx->producer_control == nullptr || acl_stream == nullptr) {
        return -1;
    }
    const auto stream = reinterpret_cast<aclrtStream>(acl_stream);
    aclError ret = aclrtMemsetAsync(ctx->protocol_status, kProtocolStatusBytes,
                                    0, kProtocolStatusBytes, stream);
    if (ret != ACL_SUCCESS) return static_cast<int>(ret);
    if (clear_wave_control == 0) return 0;
    ret = aclrtMemsetAsync(ctx->producer_control, kProducerControlBytes,
                           0, kProducerControlBytes, stream);
    return ret == ACL_SUCCESS ? 0 : static_cast<int>(ret);
}

extern "C" int mfmc2_init_credit_async(mfmc2_context_t* opaque,
                                         void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr || acl_stream == nullptr)
        return -1;
    const auto stream = reinterpret_cast<aclrtStream>(acl_stream);
    int ret = mfmc2_device_ack_async(
        ctx->layout.pool_base, ctx->layout.ack_slot, ctx->layout.peer_ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 1, stream);
    if (ret != 0) return ret;
    return mfmc2_device_quiet_async(
        ctx->layout.pool_base,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 1, stream);
}

extern "C" int mfmc2_direct_producer_async(
    mfmc2_context_t* opaque, uint64_t x, uint64_t w, uint32_t batch_index,
    uint32_t generation, void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr || x == 0 || w == 0 ||
        ctx->producer_control == nullptr || generation == 0 ||
        batch_index >= ctx->layout.max_batches) {
        return -1;
    }
    return mfmc2_device_launch_direct_producer_async(
        ctx->layout.pool_base, x, w, ctx->layout.send_arena,
        ctx->layout.peer_recv_arena, ctx->layout.batch_m, batch_index,
        generation, ctx->layout.batch_bytes,
        reinterpret_cast<uint64_t>(ctx->producer_control),
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_wait_batch_async(mfmc2_context_t* opaque,
                                        uint32_t batch_index,
                                        void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr ||
        batch_index >= ctx->layout.max_batches) {
        return -1;
    }
    return mfmc2_device_wait_batch_async(
        ctx->layout.pool_base, ctx->layout.expected_recv_base, batch_index,
        ctx->layout.batch_bytes,
        reinterpret_cast<uint64_t>(ctx->protocol_status),
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_add_batch_async(mfmc2_context_t* opaque, uint64_t out,
                                       uint32_t batch_index,
                                       uint32_t valid_rows,
                                       void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    constexpr uint32_t kAddAlignElems = 128;
    constexpr uint32_t kAddMaxBlocks = 8;
    if (ctx == nullptr || acl_stream == nullptr || out == 0 ||
        batch_index >= ctx->layout.max_batches || valid_rows == 0 ||
        valid_rows > ctx->layout.batch_m) {
        return -1;
    }
    const uint64_t elems =
        static_cast<uint64_t>(valid_rows) * kOutputWidthElems;
    if ((elems % kAddAlignElems) != 0) return -1;
    const uint32_t block_count =
        valid_rows < kAddMaxBlocks ? valid_rows : kAddMaxBlocks;
    const uint64_t batch_off =
        static_cast<uint64_t>(batch_index) * ctx->layout.batch_bytes;
    return mfmc2_device_add_async(
        out, ctx->layout.send_arena + batch_off,
        ctx->layout.recv_arena + batch_off, elems, block_count,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_ack_async(mfmc2_context_t* opaque, void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr || acl_stream == nullptr)
        return -1;
    return mfmc2_device_ack_async(
        ctx->layout.pool_base, ctx->layout.ack_slot, ctx->layout.peer_ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_gate_async(mfmc2_context_t* opaque, void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr || acl_stream == nullptr)
        return -1;
    return mfmc2_device_gate_async(
        ctx->layout.pool_base, ctx->layout.expected_credit_dst,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_quiet_async(mfmc2_context_t* opaque, void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr || acl_stream == nullptr)
        return -1;
    return mfmc2_device_quiet_async(
        ctx->layout.pool_base,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 1,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_warmup_producer_async(mfmc2_context_t* opaque,
                                             void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr) return -1;
    return mfmc2_device_warmup_producer_async(
        ctx->layout.pool_base, ctx->layout.batch_m, ctx->layout.batch_bytes,
        reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_warmup_waiter_async(mfmc2_context_t* opaque,
                                           void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr) return -1;
    return mfmc2_device_warmup_waiter_async(
        ctx->layout.pool_base, reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_warmup_add_async(mfmc2_context_t* opaque,
                                        void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || acl_stream == nullptr) return -1;
    return mfmc2_device_warmup_add_async(
        ctx->layout.send_arena, reinterpret_cast<aclrtStream>(acl_stream));
}

extern "C" int mfmc2_warmup_protocol_async(mfmc2_context_t* opaque,
                                             void* acl_stream)
{
    auto* ctx = reinterpret_cast<mfmc2_context*>(opaque);
    if (ctx == nullptr || ctx->protocol_status == nullptr || acl_stream == nullptr)
        return -1;
    const auto stream = reinterpret_cast<aclrtStream>(acl_stream);
    int ret = mfmc2_device_ack_async(
        ctx->layout.pool_base, ctx->layout.ack_slot, ctx->layout.peer_ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 0, stream);
    if (ret != 0) return ret;
    ret = mfmc2_device_gate_async(
        ctx->layout.pool_base, ctx->layout.ack_slot,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 0, stream);
    if (ret != 0) return ret;
    return mfmc2_device_quiet_async(
        ctx->layout.pool_base,
        reinterpret_cast<uint64_t>(ctx->protocol_status), 0, stream);
}
