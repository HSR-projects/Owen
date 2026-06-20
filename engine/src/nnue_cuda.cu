// Owen Engine © HSR-Projects
// SPDX-License-Identifier: GPL-3.0-or-later

#include "nnue_cuda.h"
#include <cuda_runtime.h>
#include <sstream>

namespace owen {

namespace {

constexpr int CUDA_INPUT_SIZE = 12 * 64 + 5;

std::string cuda_error(const char* action, cudaError_t err) {
    std::ostringstream ss;
    ss << action << ": " << cudaGetErrorString(err);
    return ss.str();
}

bool ok(cudaError_t err, const char* action, std::string* error) {
    if (err == cudaSuccess) return true;
    if (error) *error = cuda_error(action, err);
    return false;
}

__global__ void nnue_forward_kernel(const float* __restrict__ w1,
                                    const float* __restrict__ b1,
                                    const float* __restrict__ w2,
                                    const int* __restrict__ features,
                                    int feature_count,
                                    int hidden,
                                    float b2,
                                    float* out) {
    extern __shared__ float partial[];
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int h = tid; h < hidden; h += blockDim.x) {
        float acc = b1[h];
        const float* row = w1 + h * CUDA_INPUT_SIZE;
        for (int i = 0; i < feature_count; ++i) acc += row[features[i]];
        if (acc > 0.0f) sum += w2[h] * acc;
    }

    partial[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0) *out = b2 + partial[0];
}

} // namespace

NNUECudaBackend::~NNUECudaBackend() {
    release();
}

bool NNUECudaBackend::available() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void NNUECudaBackend::release() {
    if (d_w1_) cudaFree(d_w1_);
    if (d_b1_) cudaFree(d_b1_);
    if (d_w2_) cudaFree(d_w2_);
    if (d_features_) cudaFree(d_features_);
    if (d_out_) cudaFree(d_out_);
    d_w1_ = nullptr;
    d_b1_ = nullptr;
    d_w2_ = nullptr;
    d_features_ = nullptr;
    d_out_ = nullptr;
    hidden_ = 0;
}

bool NNUECudaBackend::upload(int hidden, const std::vector<float>& w1,
                             const std::vector<float>& b1, const std::vector<float>& w2,
                             float b2, std::string* error) {
    if (!available()) {
        if (error) *error = "no CUDA device found";
        return false;
    }

    release();
    hidden_ = hidden;
    b2_ = b2;

    const size_t w1_bytes = w1.size() * sizeof(float);
    const size_t b1_bytes = b1.size() * sizeof(float);
    const size_t w2_bytes = w2.size() * sizeof(float);

    if (!ok(cudaMalloc(&d_w1_, w1_bytes), "cudaMalloc w1", error)) return false;
    if (!ok(cudaMalloc(&d_b1_, b1_bytes), "cudaMalloc b1", error)) return false;
    if (!ok(cudaMalloc(&d_w2_, w2_bytes), "cudaMalloc w2", error)) return false;
    if (!ok(cudaMalloc(&d_features_, MAX_FEATURES * sizeof(int)), "cudaMalloc features", error)) return false;
    if (!ok(cudaMalloc(&d_out_, sizeof(float)), "cudaMalloc output", error)) return false;

    if (!ok(cudaMemcpy(d_w1_, w1.data(), w1_bytes, cudaMemcpyHostToDevice), "cudaMemcpy w1", error)) return false;
    if (!ok(cudaMemcpy(d_b1_, b1.data(), b1_bytes, cudaMemcpyHostToDevice), "cudaMemcpy b1", error)) return false;
    if (!ok(cudaMemcpy(d_w2_, w2.data(), w2_bytes, cudaMemcpyHostToDevice), "cudaMemcpy w2", error)) return false;
    return true;
}

bool NNUECudaBackend::evaluate(const std::vector<int>& features, float* out, std::string* error) const {
    if (!d_w1_ || !d_b1_ || !d_w2_ || !d_features_ || !d_out_ || hidden_ <= 0) {
        if (error) *error = "CUDA NNUE weights are not uploaded";
        return false;
    }
    if (features.size() > MAX_FEATURES) {
        if (error) *error = "too many active NNUE features";
        return false;
    }

    const int feature_count = static_cast<int>(features.size());
    if (!ok(cudaMemcpy(d_features_, features.data(), feature_count * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy features", error)) {
        return false;
    }

    int threads = 1;
    while (threads < hidden_ && threads < 256) threads <<= 1;
    nnue_forward_kernel<<<1, threads, threads * sizeof(float)>>>(
        static_cast<const float*>(d_w1_), static_cast<const float*>(d_b1_),
        static_cast<const float*>(d_w2_), static_cast<const int*>(d_features_),
        feature_count, hidden_, b2_, static_cast<float*>(d_out_));
    if (!ok(cudaGetLastError(), "launch nnue_forward_kernel", error)) return false;
    if (!ok(cudaMemcpy(out, d_out_, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy output", error)) return false;
    return true;
}

} // namespace owen
