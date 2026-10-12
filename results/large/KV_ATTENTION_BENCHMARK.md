# Python CPU vs Handwritten Metal GPU: KV Attention Benchmark

Device: **Apple M4**. Python 3.13.4.

Workload: prefix 512, 16 decode steps, hidden size 256, attention width 128; single head, batch 1, float32 GPU.

| Path | Total median (ms) | Prefill median (ms) | Decode median (ms) | Total min / p95 (ms) | Max output error |
|---|---:|---:|---:|---:|---:|
| 1. Python CPU attention, uncached | 33277.636 | 1911.265 | 31365.827 | 32602.292 / 33619.317 | 0 |
| 2. Handwritten Metal GPU attention, uncached | 17.670 | 0.976 | 16.510 | 16.298 / 19.503 | 5.66e-08 |
| 3. Handwritten Metal GPU attention, KV cached | 7.128 | 0.987 | 5.974 | 6.569 / 7.485 | 5.66e-08 |

GPU KV-cache decode speedup, completed-call timing: **2.76x**.
GPU execution-only decode: 11.935 ms without cache vs 2.414 ms with cache (**4.94x**).

KV projection rows: 8840 without cache vs 528 with cache (16.74x less KV projection work). Persistent KV capacity: 540,672 bytes.

## Measurement Boundaries

- Python repeats: 3; GPU repeats: 30; GPU warmup runs per mode: 3.
- Native initialization/compilation/buffer upload took 39.001 ms and is excluded.
- Total times include the call, completed execution, and result conversion. Inputs and weights are already prepared and GPU-resident for both GPU modes.
- Prefill/decode times measure per-step work; GPU times include encoding, submission, and waiting. GPU execution-only times use Metal command-buffer timestamps.
- Each GPU step waits for completion. Both modes use identical kernels, precision, buffers, submission counts, and attention calculations.
- Every GPU trial is checked against every prefill/decode reference output with maximum absolute error <= 1e-4. Python arithmetic uses doubles on float32 inputs.
- Pure Python uses explicit loops, not NumPy/BLAS/PyTorch. Its comparison with GPU includes interpreter and algorithm-implementation differences; it does not isolate foreign-function-call overhead or compare optimized CPU BLAS.
- All projection, score, softmax, and weighted-value calculations are handwritten. The GPU kernels use Metal Shading Language; the host bridge uses Objective-C++.
- No third-party Python packages or numerical libraries are required. System dependencies are Python, a compiler, Apple's Metal/Foundation frameworks, and the GPU driver.
- This is a small single-layer attention benchmark with predetermined token vectors, not a trained autoregressive model or a vLLM throughput benchmark.
- Cache reuse saves historical K/V projections. Attention still reads the whole retained prefix. Prefill and GPU submission overhead remain.
- A speedup below 1 means slower for this measurement. Results depend on shape, thermal state, system activity, and GPU dispatch overhead.
- JSON contains all timing samples and parameters. Re-run to assess stability.
