#include "qkv.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace {

struct DeviceBuffer {
    float* pointer = nullptr;
    ~DeviceBuffer() {
        if (pointer) cudaFree(pointer);
    }
};

__global__ void qkv_kernel(const float* input, const float* weights,
                           float* output, int tokens, int hidden, int outputs) {
    const std::size_t token =
        static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= static_cast<std::size_t>(tokens) ||
        column >= static_cast<std::size_t>(outputs)) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < hidden; ++k) {
        sum += input[token * hidden + k] * weights[column * hidden + k];
    }
    output[token * outputs + column] = sum;
}

}  // namespace

extern "C" const char* qkv_backend(void) { return "CUDA GPU"; }

extern "C" const char* qkv_error_message(int status) {
    if (status == -1) return "null pointer or nonpositive dimension";
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

extern "C" int qkv_project(const float* input, const float* weights,
                           float* output, int tokens, int hidden, int outputs) {
    if (!input || !weights || !output || tokens <= 0 || hidden <= 0 ||
        outputs <= 0) {
        return -1;
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(tokens) * hidden * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(outputs) * hidden * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(tokens) * outputs * sizeof(float);
    DeviceBuffer device_input, device_weights, device_output;

    // Early returns release any buffers already allocated through RAII.
#define CHECK_CUDA(call)                         \
    do {                                         \
        const cudaError_t status = (call);        \
        if (status != cudaSuccess) return status; \
    } while (false)

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_input.pointer),
                          input_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_weights.pointer),
                          weight_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_output.pointer),
                          output_bytes));
    CHECK_CUDA(cudaMemcpy(device_input.pointer, input, input_bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_weights.pointer, weights, weight_bytes,
                          cudaMemcpyHostToDevice));

    const dim3 threads(16, 16);
    const dim3 blocks((static_cast<unsigned int>(outputs) + 15) / 16,
                      (static_cast<unsigned int>(tokens) + 15) / 16);
    qkv_kernel<<<blocks, threads>>>(device_input.pointer, device_weights.pointer,
                                   device_output.pointer, tokens, hidden, outputs);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(output, device_output.pointer, output_bytes,
                          cudaMemcpyDeviceToHost));
#undef CHECK_CUDA
    return 0;
}
