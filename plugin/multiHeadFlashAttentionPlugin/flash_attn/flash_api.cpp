/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
// #include <torch/python.h>
// #include <torch/nn/functional.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <iostream>

#include "src/flash.h"
#include "src/static_switch.h"

#include <cutlass/numeric_types.h>
#include <cuda_runtime.h>

// #define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
// #define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
namespace nvinfer1
{
namespace plugin
{

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void* q,
                      void* k,
                      void* v,
                      void* out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      bool is_causal) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = false;

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = h * d;
    params.k_row_stride = h_k * d;
    params.v_row_stride = h_k * d;
    params.q_head_stride = d;
    params.k_head_stride = d;
    params.v_head_stride = d;
    params.o_ptr = out;
    params.o_row_stride = h * d;
    params.o_head_stride = d;

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = seqlen_q * h * d;
        params.k_batch_stride = seqlen_k * h_k * d;
        params.v_batch_stride = seqlen_k * h_k * d;
        params.o_batch_stride = seqlen_q * h * d;
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    // TORCH_CHECK(p_dropout < 1.f);

    params.is_causal = is_causal;
    params.is_seqlens_k_cumulative = true;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        FWD_HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            }
        });
    });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

int32_t
mha_fwd(void const* q,         // batch_size x seqlen_q x num_heads x head_size
        void const* k,         // batch_size x seqlen_k x num_heads_k x head_size
        void const* v,         // batch_size x seqlen_k x num_heads_k x head_size
        void const* out_,      // batch_size x seqlen_q x num_heads x head_size
        void const* softmax_lse,
        int32_t batch_size,
        int32_t num_heads,
        int32_t num_heads_k,
        int32_t head_size_og, 
        int32_t seqlen_q,
        int32_t seqlen_k,
        cudaStream_t stream) {

    int device;
    FMHA_CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp dprops;
    FMHA_CHECK_CUDA(cudaGetDeviceProperties(&dprops, device));
    
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops.major == 8 && dprops.minor >= 0;
    bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
    // TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, at::kFloat);

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     const_cast<void*>(q),
                     const_cast<void*>(k),
                     const_cast<void*>(v),
                     const_cast<void*>(out_),
                     nullptr,
                     nullptr,
                     nullptr,
                     const_cast<void*>(softmax_lse),
                     0.0f,
                     sqrt(head_size_og),
                     false);

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = is_sm90 || is_sm8x
        ? (head_size <= 64 ? 256 : (head_size <= 160 ? 128 : 64))
        : (head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64));
    const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
    params.num_splits = 1;
    // params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount, num_n_blocks, 128);
    // if (params.num_splits > 1) {
    //     at::Tensor softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    //     at::Tensor out_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
    //     params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
    //     params.oaccum_ptr = out_accum.data_ptr();
    // }

    run_mha_fwd(params, stream);

    return 0;
}

} // name space plugin
} // namespace nvinfer1
