#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MFMC2_ADAPTER_ABI_VERSION 1u
#define MFMC2_MAX_BATCHES 64u
#define MFMC2_COOPERATIVE_CORES 8u

typedef struct mfmc2_context mfmc2_context_t;

typedef struct mfmc2_layout {
    uint64_t pool_base;
    uint64_t own_segment;
    uint64_t peer_segment;
    uint64_t symmetric_size;
    uint64_t local_size;

    uint64_t send_arena;
    uint64_t recv_arena;
    uint64_t peer_recv_arena;
    uint64_t ack_slot;
    uint64_t peer_ack_slot;

    uint64_t arena_bytes;
    uint64_t batch_bytes;
    uint64_t expected_credit_dst;
    uint64_t expected_recv_base;

    uint32_t arena_rows;
    uint32_t batch_m;
    uint32_t max_batches;
    uint32_t reserved;
} mfmc2_layout_t;

uint32_t mfmc2_adapter_abi_version(void);

int mfmc2_create(int rank,
                 int world_size,
                 const char* store_url,
                 uint64_t local_size,
                 uint32_t arena_rows,
                 uint32_t batch_m,
                 mfmc2_context_t** out_ctx);
int mfmc2_destroy(mfmc2_context_t* ctx);
int mfmc2_get_layout(mfmc2_context_t* ctx, mfmc2_layout_t* out_layout);
int mfmc2_debug_protocol_status(mfmc2_context_t* ctx, uint64_t* out_status);

int mfmc2_control_barrier(mfmc2_context_t* ctx);
int mfmc2_exchange_geometry(mfmc2_context_t* ctx);

int mfmc2_prepare_wave_async(mfmc2_context_t* ctx,
                             uint32_t clear_wave_control,
                             void* acl_stream);
int mfmc2_init_credit_async(mfmc2_context_t* ctx, void* acl_stream);

int mfmc2_direct_producer_async(mfmc2_context_t* ctx,
                                uint64_t x,
                                uint64_t w,
                                uint32_t batch_index,
                                uint32_t generation,
                                void* acl_stream);
int mfmc2_wait_batch_async(mfmc2_context_t* ctx,
                           uint32_t batch_index,
                           void* acl_stream);
int mfmc2_add_batch_async(mfmc2_context_t* ctx,
                          uint64_t out,
                          uint32_t batch_index,
                          uint32_t valid_rows,
                          void* acl_stream);

int mfmc2_ack_async(mfmc2_context_t* ctx, void* acl_stream);
int mfmc2_gate_async(mfmc2_context_t* ctx, void* acl_stream);
int mfmc2_quiet_async(mfmc2_context_t* ctx, void* acl_stream);

int mfmc2_warmup_producer_async(mfmc2_context_t* ctx, void* acl_stream);
int mfmc2_warmup_waiter_async(mfmc2_context_t* ctx, void* acl_stream);
int mfmc2_warmup_add_async(mfmc2_context_t* ctx, void* acl_stream);
int mfmc2_warmup_protocol_async(mfmc2_context_t* ctx, void* acl_stream);

#ifdef __cplusplus
}
#endif
