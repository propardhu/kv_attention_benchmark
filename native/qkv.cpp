#include "qkv.h"

#include <cstddef>

extern "C" const char* qkv_backend(void) { return "C++ CPU"; }

extern "C" const char* qkv_error_message(int status) {
    return status == 0 ? "success" : "null pointer or nonpositive dimension";
}

extern "C" int qkv_project(const float* input, const float* weights,
                           float* output, int tokens, int hidden, int outputs) {
    if (!input || !weights || !output || tokens <= 0 || hidden <= 0 ||
        outputs <= 0) {
        return -1;
    }

    for (int token = 0; token < tokens; ++token) {
        for (int column = 0; column < outputs; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                sum += input[static_cast<std::size_t>(token) * hidden + k] *
                       weights[static_cast<std::size_t>(column) * hidden + k];
            }
            output[static_cast<std::size_t>(token) * outputs + column] = sum;
        }
    }
    return 0;
}
