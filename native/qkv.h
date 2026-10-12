#pragma once

#ifdef __cplusplus
extern "C" {
#endif

const char* qkv_backend(void);
const char* qkv_error_message(int status);

// Row-major input [tokens, hidden] and weights [outputs, hidden].
// Output is input @ weights.T, stored as [tokens, outputs].
int qkv_project(const float* input, const float* weights, float* output,
                int tokens, int hidden, int outputs);

#ifdef __cplusplus
}
#endif
