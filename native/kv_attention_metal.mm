#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <string>

static thread_local std::string last_error;

static NSString* kernel_source = @R"METAL(
#include <metal_stdlib>
using namespace metal;
struct Params { uint hidden, width, start, count, length, step; };

kernel void project_q_and_kv(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device float* qkv [[buffer(2)]],
                    constant Params& p [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
    uint columns = 3 * p.width;
    if (id >= p.count * columns) return;
    uint token = p.start + id / columns;
    uint column = id % columns;
    if (column < p.width && token != p.length - 1) return;
    float sum = 0;
    for (uint k = 0; k < p.hidden; ++k)
        sum += x[token * p.hidden + k] * w[column * p.hidden + k];
    if (column < p.width) qkv[column] = sum;
    else qkv[p.width + token * 2 * p.width + column - p.width] = sum;
}

kernel void compute_attention_scores(device const float* qkv [[buffer(0)]],
                  device float* scores [[buffer(1)]],
                  constant Params& p [[buffer(2)]],
                  uint token [[thread_position_in_grid]]) {
    if (token >= p.length) return;
    float sum = 0;
    for (uint d = 0; d < p.width; ++d)
        sum += qkv[d] * qkv[p.width + token * 2 * p.width + d];
    scores[token] = sum / sqrt(float(p.width));
}

kernel void apply_attention_weights(device const float* qkv [[buffer(0)]],
                   device const float* scores [[buffer(1)]],
                   device float* output [[buffer(2)]],
                   constant Params& p [[buffer(3)]],
                   uint d [[thread_position_in_grid]]) {
    if (d >= p.width) return;
    float maximum = scores[0];
    for (uint t = 1; t < p.length; ++t) maximum = max(maximum, scores[t]);
    float denominator = 0, numerator = 0;
    for (uint t = 0; t < p.length; ++t) {
        float probability = exp(scores[t] - maximum);
        denominator += probability;
        numerator += probability * qkv[p.width + t * 2 * p.width + p.width + d];
    }
    output[p.step * p.width + d] = numerator / denominator;
}
)METAL";

struct Params { uint32_t hidden, width, start, count, length, step; };
struct Attention {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> projection, score, attend;
    id<MTLBuffer> inputs, weights, qkv, scores, output;
    uint32_t prefix, steps, hidden, width;
};

extern "C" const char* attention_error() { return last_error.c_str(); }

extern "C" void* attention_create(const float* x, const float* w,
                                   int prefix, int steps, int hidden, int width) {
    @autoreleasepool {
        if (!x || !w || prefix <= 0 || steps < 0 || hidden <= 0 || width <= 0) {
            last_error = "Invalid input pointers or dimensions";
            return nullptr;
        }
        Attention* a = new Attention{};
        a->device = MTLCreateSystemDefaultDevice();
        if (!a->device) {
            last_error = "No Metal GPU available to this process";
            delete a;
            return nullptr;
        }
        a->queue = [a->device newCommandQueue];
        NSError* error = nil;
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [a->device newLibraryWithSource:kernel_source
                                                      options:options error:&error];
        if (!library) {
            last_error = error.localizedDescription.UTF8String;
            delete a;
            return nullptr;
        }
        auto pipeline = [&](NSString* name) {
            return [a->device newComputePipelineStateWithFunction:
                [library newFunctionWithName:name] error:&error];
        };
        a->projection = pipeline(@"project_q_and_kv");
        a->score = pipeline(@"compute_attention_scores");
        a->attend = pipeline(@"apply_attention_weights");
        a->prefix = prefix; a->steps = steps; a->hidden = hidden; a->width = width;
        NSUInteger total = static_cast<NSUInteger>(prefix) + steps;
        a->inputs = [a->device newBufferWithBytes:x length:total * hidden * sizeof(float)
                                         options:MTLResourceStorageModeShared];
        a->weights = [a->device newBufferWithBytes:w length:3 * width * hidden * sizeof(float)
                                          options:MTLResourceStorageModeShared];
        a->qkv = [a->device newBufferWithLength:(total * 2 + 1) * width * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        a->scores = [a->device newBufferWithLength:total * sizeof(float)
                                        options:MTLResourceStorageModeShared];
        a->output = [a->device newBufferWithLength:(steps + 1) * width * sizeof(float)
                                        options:MTLResourceStorageModeShared];
        if (!a->queue || !a->projection || !a->score || !a->attend || !a->inputs ||
            !a->weights || !a->qkv || !a->scores || !a->output) {
            last_error = "Failed to create Metal pipelines or buffers";
            delete a;
            return nullptr;
        }
        return a;
    }
}

static void dispatch(id<MTLComputeCommandEncoder> encoder,
                     id<MTLComputePipelineState> pipeline, NSUInteger count) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger threads = std::min<NSUInteger>(64, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

extern "C" int attention_run(void* handle, int cached,
                              double* step_seconds, double* gpu_seconds) {
    @autoreleasepool {
        if (!handle || !step_seconds || !gpu_seconds) return -1;
        Attention* a = static_cast<Attention*>(handle);
        for (uint32_t step = 0; step <= a->steps; ++step) {
            auto start = std::chrono::steady_clock::now();
            uint32_t length = a->prefix + step;
            Params p{a->hidden, a->width, cached && step ? length - 1 : 0,
                     cached && step ? 1 : length, length, step};
            id<MTLCommandBuffer> command = [a->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setBuffer:a->inputs offset:0 atIndex:0];
            [encoder setBuffer:a->weights offset:0 atIndex:1];
            [encoder setBuffer:a->qkv offset:0 atIndex:2];
            [encoder setBytes:&p length:sizeof(p) atIndex:3];
            dispatch(encoder, a->projection, p.count * 3 * p.width);
            [encoder endEncoding];

            encoder = [command computeCommandEncoder];
            [encoder setBuffer:a->qkv offset:0 atIndex:0];
            [encoder setBuffer:a->scores offset:0 atIndex:1];
            [encoder setBytes:&p length:sizeof(p) atIndex:2];
            dispatch(encoder, a->score, length);
            [encoder endEncoding];

            encoder = [command computeCommandEncoder];
            [encoder setBuffer:a->qkv offset:0 atIndex:0];
            [encoder setBuffer:a->scores offset:0 atIndex:1];
            [encoder setBuffer:a->output offset:0 atIndex:2];
            [encoder setBytes:&p length:sizeof(p) atIndex:3];
            dispatch(encoder, a->attend, a->width);
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                last_error = command.error.localizedDescription.UTF8String;
                return -1;
            }
            step_seconds[step] = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            gpu_seconds[step] = command.GPUEndTime - command.GPUStartTime;
        }
        return 0;
    }
}

extern "C" const float* attention_output(void* handle) {
    return static_cast<const float*>(static_cast<Attention*>(handle)->output.contents);
}

extern "C" const char* attention_device(void* handle) {
    return static_cast<Attention*>(handle)->device.name.UTF8String;
}

extern "C" void attention_destroy(void* handle) { delete static_cast<Attention*>(handle); }
