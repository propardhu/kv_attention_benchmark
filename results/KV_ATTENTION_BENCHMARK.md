# Python CPU vs Handwritten Metal GPU: KV Attention Benchmark

Device: **Apple M4**. Python 3.13.4.

Workload: prefix 256, 16 decode steps, hidden size 128, attention width 64; single head, batch 1, float32 GPU.

| Path | Total median (ms) | Prefill median (ms) | Decode median (ms) | Total min / p95 (ms) | Max output error |
|---|---:|---:|---:|---:|---:|
| 1. Python CPU attention, uncached | 4093.843 | 231.156 | 3866.877 | 4091.530 / 4122.649 | 0 |
| 2. Handwritten Metal GPU attention, uncached | 5.661 | 0.320 | 5.273 | 5.252 / 9.575 | 5.08e-08 |
| 3. Handwritten Metal GPU attention, KV cached | 5.028 | 0.329 | 4.632 | 4.536 / 5.728 | 5.08e-08 |

GPU KV-cache decode speedup, completed-call timing: **1.14x**.
GPU execution-only decode: 1.845 ms without cache vs 1.229 ms with cache (**1.50x**).

KV projection rows: 4488 without cache vs 272 with cache (16.50x less KV projection work). Persistent KV capacity: 139,264 bytes.

## Measurement Boundaries

- Labels were updated to describe the implementations accurately; the previously measured timing samples are unchanged.
- All projection, score, softmax, and weighted-value calculations are handwritten. The GPU kernels use Metal Shading Language; the host bridge uses Objective-C++.
- No third-party Python packages or numerical libraries are required. System dependencies are Python, a compiler, Apple's Metal/Foundation frameworks, and the GPU driver.

- Python repeats: 3; GPU repeats: 30; GPU warmup runs per mode: 3.
- Native initialization/compilation/buffer upload took 29.981 ms and is excluded.
- Total times include the call, completed execution, and result conversion. Inputs and weights are already prepared and GPU-resident for both GPU modes.
- Prefill/decode times measure per-step work; GPU times include encoding, submission, and waiting. GPU execution-only times use Metal command-buffer timestamps.
- Each GPU step waits for completion. Both modes use identical kernels, precision, buffers, submission counts, and attention calculations.
- Every GPU trial is checked against every prefill/decode reference output with maximum absolute error <= 1e-4. Python arithmetic uses doubles on float32 inputs.
- Pure Python uses explicit loops, not NumPy/BLAS/PyTorch. Its comparison with GPU includes interpreter and algorithm-implementation differences; it does not isolate foreign-function-call overhead or compare optimized CPU BLAS.
- This is a small single-layer attention benchmark with predetermined token vectors, not a trained autoregressive model or a vLLM throughput benchmark.
- Cache reuse saves historical K/V projections. Attention still reads the whole retained prefix. Prefill and GPU submission overhead remain.
- A speedup below 1 means slower for this measurement. Results depend on shape, thermal state, system activity, and GPU dispatch overhead.
- JSON contains all timing samples and parameters. Re-run to assess stability.
