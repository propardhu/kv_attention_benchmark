# KV Attention Benchmark

An educational benchmark showing how Python calls handwritten native GPU code,
how single-head attention works, and how retaining keys and values avoids
repeated projection work during decoding.

The main benchmark compares a Python CPU reference with Apple Metal GPU
attention, with and without KV reuse. A separate C++ CPU QKV demo and optional
CUDA QKV example illustrate the native-call boundary with a smaller operation.
There are no third-party Python package dependencies.

This is a source-level learning project. Its deterministic input vectors stand
in for already embedded tokens. It does not load a trained model, tokenize text,
generate tokens, or implement a full inference engine.

## Contents

- [Build and run](#three-way-efficiency-experiment)
- [Recorded measurements](#recorded-measurements)
- [Arguments and matrix shapes](#arguments-and-matrix-shapes)
- [KV reuse and memory](#kv-reuse-and-memory)
- [Implementation and measurement](#implementation-and-measurement)
- [Dependencies](#what-we-wrote-and-what-we-depend-on)
- [Original QKV demo](#original-qkv-projection-demo)
- [Troubleshooting](#troubleshooting)

## Three-Way Efficiency Experiment

The main experiment now compares the same single-head attention workload:

1. **Python CPU attention, uncached:** explicit loops compute K/V
   projections for the whole prefix at every step, then attention for the last token.
2. **Handwritten Metal GPU attention, uncached:** Python calls an Objective-C++ shared
   library through `ctypes`; Metal GPU kernels perform the same computation.
3. **Handwritten Metal GPU attention, KV cached:** the same kernels keep historical K/V
   on the GPU and project only the newly appended token after prefill.

All three compute `softmax(Q @ K.T / sqrt(head_width)) @ V` for the current last
token. Q is computed for that token only. This is an attention workload built
from matrix operations, rather than caching results of an arbitrary standalone
matrix multiplication. Cache mode retains K and V, not old queries.

### Run on This Mac

```bash
cd kv_attention_benchmark
python3 build.py --backend cpu
python3 build.py --backend metal
ATTENTION_TEST_GPU=1 python3 -m unittest -v
python3 benchmark_kv_attention.py
```

Metal GPU access may require allowing the command outside a restricted sandbox.
The `.mm` file contains Objective-C++ host code that calls Apple's Metal API.
Its embedded Metal Shading Language source is compiled at initialization into
GPU pipelines. This uses the Apple GPU, not CUDA. The original optional CUDA
QKV example below remains available, but it does not implement this three-way
attention benchmark on NVIDIA hardware.

Results are saved to [results/KV_ATTENTION_BENCHMARK.md](results/KV_ATTENTION_BENCHMARK.md)
and [results/kv_attention_benchmark.json](results/kv_attention_benchmark.json). The JSON
records every measurement, parameters, output errors, and setup time.

For a different shape:

```bash
python3 benchmark_kv_attention.py --prefix 128 --steps 8 --hidden 64 --width 32 --output results/small
```

The project folder is the working directory for the remaining commands. On a
fresh clone, build the CPU backend before running the full test suite because
the original QKV tests load its native library. Build artifacts and Python
bytecode are ignored by Git; measured reports and raw timing samples are included.

## Recorded Measurements

Recorded on an **Apple M4**, Python 3.13.4. Times below are medians for a complete
workload: one prefill and 16 decode steps.

| Prefix / hidden / attention width | Python CPU uncached | Metal GPU uncached | Metal GPU KV cached | GPU decode speedup from KV reuse |
|---|---:|---:|---:|---:|
| 256 / 128 / 64 | 4,093.84 ms | 5.66 ms | 5.03 ms | 1.14x |
| 512 / 256 / 128 | 33,277.64 ms | 17.67 ms | 7.13 ms | 2.76x |

In the larger experiment, GPU execution-only decode time fell from 11.935 ms to
2.414 ms, a 4.94x improvement. Completed decode wall time fell from 16.510 ms to
5.974 ms: encoding, submission, and waiting reduce the observed gain. Maximum
output error against the Python reference was about `5.66e-8`.

Read the [default report](results/KV_ATTENTION_BENCHMARK.md) and
[larger-workload report](results/large/KV_ATTENTION_BENCHMARK.md) for distributions
and limits. Their JSON files record each trial, dimensions, setup time, device,
and numerical error. Results depend on hardware, thermal state, and system load.

The matched GPU-to-GPU comparison is the useful evidence for KV reuse. The
Python-to-GPU difference includes interpreter overhead and implementation
differences; it does not isolate the benefit of foreign-function calls or compare
the GPU with an optimized CPU BLAS implementation.

## Arguments and Matrix Shapes

Reproduce the larger experiment:

```bash
python3 benchmark_kv_attention.py \
  --prefix 512 \
  --steps 16 \
  --hidden 256 \
  --width 128 \
  --python-repeats 3 \
  --gpu-repeats 30 \
  --output results/large
```

| Argument | Default | Meaning |
|---|---:|---|
| `--prefix` | 256 | Initial number of token vectors |
| `--steps` | 16 | Additional tokens processed one at a time |
| `--hidden` | 128 | Input features per token |
| `--width` | 64 | Features in Q, K, and V; one attention head |
| `--python-repeats` | 3 | Complete Python reference trials |
| `--gpu-repeats` | 30 | Complete trials per GPU mode |
| `--warmup` | 3 | Untimed trials per GPU mode |
| `--output` | `results` | Destination for Markdown and JSON reports |

Dimensions and repeat counts must be positive; warmup must be at least one.
The GPU bridge limits dimensions to 65,535 as an implementation guard, not a
recommended workload size. The explicit-loop Python baseline becomes expensive
quickly. A run overwrites report files in its chosen output directory; use a
different directory to preserve an earlier experiment.

Let `P` be prefix length, `T` decode steps, `H` hidden size, and `D` attention
width. At step `s`, context length is `L = P + s`. There are `T + 1` evaluations:
one prefill and `T` decode steps. Saved output vectors have shape `[T + 1, D]`.

### Projection Shapes

```text
X:                  [L, H]
Wq, Wk, Wv:         [D, H] each
Combined weights:   [3D, H]

Q_current = X_last @ Wq.T       -> [1, D]
K         = X      @ Wk.T       -> [L, D]
V         = X      @ Wv.T       -> [L, D]
```

Only the current last token's Q is computed. The kernels perform these products
with explicit dot-product loops rather than a library GEMM call. There is no
bias, positional encoding, or multi-head split.

| Matrix | Larger workload at prefill | Larger workload at final decode |
|---|---:|---:|
| Available input X | `512 x 256` | `528 x 256` |
| Combined weights | `384 x 256` | `384 x 256` |
| Current Q | `1 x 128` | `1 x 128` |
| Retained K and V, each | `512 x 128` | `528 x 128` |
| Attention scores | `1 x 512` | `1 x 528` |
| Attention output | `1 x 128` | `1 x 128` |

### Attention Shapes

```text
scores        = Q_current @ K.T / sqrt(D)  -> [1, L]
probabilities = softmax(scores)            -> [1, L]
output        = probabilities @ V          -> [1, D]

At prefill:
(1 x 128) @ (128 x 512) = 1 x 512 scores
(1 x 512) @ (512 x 128) = 1 x 128 output

At final decode:
(1 x 128) @ (128 x 528) = 1 x 528 scores
(1 x 528) @ (528 x 128) = 1 x 128 output
```

Softmax subtracts the largest score before exponentiation for stability. The
current last token can attend to every available token, including itself;
future tokens are not part of the computation.

## KV Reuse and Memory

Without caching, each evaluation projects every available token into K/V. With
caching, prefill builds historical K/V and each decode step projects one new
token. The same projected historical vectors remain valid for these fixed
inputs and weights; old queries are not retained.

```text
Prefill:   project P tokens into K/V; compute the current query
Decode 1:  reuse P K/V pairs; append one new K/V pair
Decode 2:  reuse P+1 K/V pairs; append one new K/V pair
...
Decode T:  reuse P+T-1 K/V pairs; append one new K/V pair
```

K/V projection visits this many token rows:

```text
Uncached: (T + 1) * P + T * (T + 1) / 2
Cached:   P + T
```

With `P=512` and `T=16`, that is **8,840 versus 528 rows**, or 16.74x less K/V
projection work. It is not 16.74x less total work: attention still reads the
entire retained context at every step, and query projection still runs.

Float32 KV capacity is `2 * (P + T) * D * 4` bytes. The larger experiment uses
540,672 bytes, or 528 KiB, for K/V capacity. This excludes inputs, weights,
query, scores, output, and runtime allocations. Both GPU modes allocate the
same capacity to avoid timing differences from allocation size.

This is within-sequence KV reuse. Cross-request prefix caching, eviction,
paging, hash lookup, and distributed KV transfer are not implemented.

## Implementation and Measurement

The native shared buffer layout is:

```text
Current Q[D] | K_0[D] V_0[D] | K_1[D] V_1[D] | ...
```

| Kernel | Work |
|---|---|
| `project_q_and_kv` | Weight-row dot products; writes current Q and selected K/V rows |
| `compute_attention_scores` | One scaled query-key dot product per retained token |
| `apply_attention_weights` | Stable softmax and weighted-V reduction per output feature |

Python's `ctypes` bridge loads exported C ABI functions for context creation,
execution, output access, and cleanup. `argtypes` and `restype` define the binary
contract. The host uses Objective-C++ to allocate Metal buffers and dispatch
the three kernels in ordered encoders within a command buffer. It waits for
completion before reading output. The Python wrapper closes the native context.

The Metal kernels are embedded in the `.mm` source and compiled by Metal at
initialization. `clang++` compiles the host into
`build/libkv_attention_metal.dylib`. This approach does not require a separate
`metal` command-line compiler. Shared Metal storage is accessible by CPU and GPU
on Apple Silicon; inputs and weights are copied into it once during setup.

The script prepares identical seeded float32 inputs, warms up both GPU modes,
then alternates their order across trials. Every trial rebuilds the cache during
prefill, so it cannot reuse cached state from a previous trial.

| Timing | Includes | Excludes |
|---|---|---|
| Setup | Library/context creation, kernel compilation, buffers, upload | Attention trials |
| Total wall | Workload call, completed work, output conversion | Shared input preparation and GPU setup |
| Prefill/decode wall | Per-step work; GPU encoding/submission/waiting | Result conversion outside native execution |
| GPU execution-only | Command-buffer GPU duration | Most CPU submission and result-conversion costs |

Statistics include median, minimum, maximum, and nearest-rank p95. With three
Python samples, p95 is the largest trial. JSON retains the individual samples.
The benchmark checks every GPU output against the Python reference, rejecting
shape mismatches, nonfinite values, or absolute error above `1e-4`. Timings must
also be positive and finite. Python arithmetic uses doubles on float32 inputs;
the GPU uses float32, so small rounding differences are expected.

The kernels are deliberately simple: no shared-memory tiling, optimized GEMM
library, multi-head attention, or fusion. Small workloads can be dominated by
GPU dispatch costs. Results illustrate this implementation and cannot establish
universal GPU, KV-cache, or vLLM speedups.

### Project Structure

```text
kv_attention_benchmark/
  benchmark_kv_attention.py     CLI, correctness checks, timings, reports
  attention_workload.py         Python reference and native Metal bridge
  build.py                      CPU / Metal / optional CUDA compilation
  native/
    kv_attention_metal.mm       Objective-C++ host and Metal GPU kernels
    qkv.h                       Original projection interface
    qkv.cpp                     Original C++ CPU projection
    qkv.cu                      Optional CUDA projection
  demo.py                       Small original QKV example
  bridge.py                     Original QKV ctypes bridge
  test_kv_attention.py          Reference and opt-in GPU tests
  test_bridge.py                Original QKV tests
  results/                      Markdown reports and raw JSON samples
    large/                      Larger experiment
  build/                        Generated libraries, ignored by Git
```

### Read the New Files

| File | Purpose |
|---|---|
| [benchmark_kv_attention.py](benchmark_kv_attention.py) | Warmup, alternating GPU runs, correctness checks, timing/report generation |
| [attention_workload.py](attention_workload.py) | Handwritten Python CPU reference and the Metal `ctypes` bridge |
| [native/kv_attention_metal.mm](native/kv_attention_metal.mm) | Handwritten Objective-C++ host code and Metal GPU kernels |
| [test_kv_attention.py](test_kv_attention.py) | Reference tests and opt-in real GPU comparisons |

The native call chain is:

```text
MetalAttentionBackend.run(cached)
  -> ctypes: attention_run()
  -> Objective-C++ encodes three kernels into a Metal command buffer
  -> GPU: project K/V and current Q -> attention scores -> weighted values
  -> host waits for completion
  -> Python reads the output vectors
```

Input vectors are fixed ahead of time and are identical across modes. They stand
in for embeddings; no tokenizer or trained model is involved. The cache is
rebuilt during prefill on every trial, then reused only within that trial.
Weights, input buffers, and KV capacity are allocated once before timing;
compilation and upload are reported as separate setup costs. Both GPU paths use
the same resident buffers, dispatch count, and float32 precision. Python uses
double arithmetic on the same float32 input values, checked with a tolerance.

The baseline has **no explicit native callout in its computation**, although
CPython itself is native software. It uses no NumPy, BLAS, or PyTorch.
Consequently, CPU-to-GPU speedup measures this reference implementation versus
these kernels, not pure callout overhead or an optimized CPU library comparison.
The matched GPU-to-GPU comparison isolates KV reuse much more closely.

Reported total wall time includes completed computation and output conversion.
Separate prefill/decode wall times include GPU encoding, submission, and waiting.
Metal command-buffer timestamps report GPU execution duration, excluding much
of that host overhead. Tiny workloads can be dominated by submission costs, so
less arithmetic does not guarantee proportionally less completed-call time.

For prefix length P and T added tokens, uncached K/V projection visits
`sum(P + s for s in range(T + 1))` token rows. Cached projection visits `P + T`.
Both still calculate attention against the entire available prefix at every
step. Persistent float32 KV capacity is `2 * (P + T) * width * 4` bytes.

These are deliberately simple, single-head kernels. They do not use optimized
GEMM libraries or represent a full transformer. The useful evidence is output
equivalence, avoided projection work, and measured latency for this workload;
results cannot establish universal GPU or vLLM speedups.

### What We Wrote and What We Depend On

We wrote the computational implementations in this POC: the Python CPU
projection and attention loops, the C++ QKV matrix-multiplication loop, the CUDA
QKV kernel, the Metal attention kernels, the native dispatch code, and the
`ctypes` bridges. The Metal kernels are `project_q_and_kv`,
`compute_attention_scores`, and `apply_attention_weights`. They calculate
projections, dot products, stable softmax, and weighted values directly using
loops and scalar arithmetic. We do not delegate matrix multiplication or
attention to an external numerical library.

**There are no third-party Python package dependencies. There are system
dependencies.** GPU access and native compilation require platform software:

| Component | Required dependencies |
|---|---|
| Python reference / bridge / tests | Python 3 standard library, including `ctypes` |
| C++ CPU QKV demo | C++ compiler and standard C++ runtime |
| Metal attention benchmark | macOS 15+, Apple Metal-capable GPU, command-line compiler/SDK, Metal and Foundation system frameworks, GPU driver |
| Optional CUDA QKV demo | Linux, NVIDIA GPU/driver, CUDA toolkit/runtime including `nvcc` |

No PyTorch, NumPy, Triton, BLAS, cuBLAS, MPSGraph, or Metal Performance Shaders
attention/matrix-multiplication routine is used. Metal and CUDA provide device
memory, compilation/launch, synchronization, and timing infrastructure. Their
GPU drivers and compilers are supplied by the platform; we did not write them.
GPU `exp` and `sqrt` use the shading language's built-in math functions.

Only the Metal attention benchmark and C++ CPU demo have been executed on this
Mac. The optional CUDA QKV code was written here but has not been compiled or
tested on NVIDIA hardware. It implements projection only, without KV caching.

## Original QKV Projection Demo

This small project demonstrates calling compiled native code from Python using
only the Python standard library and a C++ compiler. Its operation mirrors the
combined QKV projection we examined in vLLM:

```text
Already-embedded token vectors X [tokens, hidden]
  -> Python packs float32 buffers
  -> ctypes calls an exported native function
  -> native implementation computes X @ W.T
  -> Python receives combined QKV [tokens, q_size + k_size + v_size]
  -> Python splits the columns into Q, K, and V
```

The CPU backend runs on macOS and Linux. The optional CUDA backend builds on
Linux with the NVIDIA CUDA toolkit and requires a compatible NVIDIA GPU and
driver to execute. CUDA cannot run on this Apple Silicon Mac.

## Run Locally

From the repository root:

```bash
cd kv_attention_benchmark
python3 build.py
python3 demo.py
python3 -m unittest -v
```

No pip dependencies are needed. `build.py` finds `clang++` or `g++` and produces
`build/libqkv_cpu.dylib` on macOS or `build/libqkv_cpu.so` on Linux.

Expected demo output:

```text
Python: owns input [2, 3] and weights [6, 3] float buffers
Bridge: ctypes -> shared library -> C++ CPU
Native result X @ W.T: [[1.0, 2.0, 3.0, 3.0, 5.0, 4.0], [4.0, 5.0, 6.0, 9.0, 11.0, 10.0]]
Python splits the combined result:
Q: [[1.0, 2.0], [4.0, 5.0]]
K: [[3.0, 3.0], [6.0, 9.0]]
V: [[5.0, 4.0], [11.0, 10.0]]
```

## Files and Reading Order

| File | Purpose |
|---|---|
| [demo.py](demo.py) | Python orchestrator: inputs, call, QKV split, output |
| [bridge.py](bridge.py) | Loads the library, declares ABI types, owns float buffers |
| [native/qkv.h](native/qkv.h) | Shared declaration of the native interface |
| [native/qkv.cpp](native/qkv.cpp) | CPU implementation of matrix multiplication |
| [native/qkv.cu](native/qkv.cu) | Optional host CUDA code and GPU kernel |
| [build.py](build.py) | Compiles one backend into a shared library |
| [test_bridge.py](test_bridge.py) | Checks numerical results, shapes, and invalid calls |

## How the Language Boundary Works

1. The compiler turns `.cpp` or `.cu` into a shared library.
2. `ctypes.CDLL(...)` loads that library into the Python process.
3. `argtypes` and `restype` describe the function's binary calling contract.
4. Python creates contiguous `ctypes.c_float` arrays and passes their pointers.
5. Native code reads the input pointers and writes into the output buffer.
6. The synchronous function returns a status; Python checks it and reads results.

`extern "C"` gives exported functions C linkage so the bridge can look them up
by predictable names. The implementation is still C++. A stable ABI requires
both sides to agree on types, dimensions, layout, ownership, and lifetime.

The CPU implementation reads Python-owned native buffers directly without an
extra native-side copy. Packing Python lists into float buffers and converting
the result back into lists do allocate/copy data. No Python objects enter the
native function; it receives pointers and integers.

The three dimensions are `tokens`, `hidden`, and `outputs`. Both input and
weights use row-major storage. Weights contain one row per output feature, so
the computation is `output[token, column] = sum_k X[token, k] * W[column, k]`.

The wrapper checks dimensions and buffer lengths before calling native code.
The native interface also checks null pointers and nonpositive dimensions, but
a raw C pointer cannot describe its allocation length. Direct native callers
must supply correctly sized, nonoverlapping buffers that remain alive for the
entire call.

## Optional NVIDIA CUDA Run

On a Linux machine with `nvcc` and a compatible NVIDIA GPU:

```bash
python3 build.py --backend cuda
python3 demo.py --backend cuda
QKV_TEST_BACKEND=cuda python3 -m unittest -v
```

The Python ABI stays identical; the CUDA shared library implements the same
exported functions. In `qkv.cu`:

```text
CPU host function: qkv_project()
  -> cudaMalloc: allocate GPU buffers
  -> cudaMemcpy: copy input and weights CPU -> GPU
  -> qkv_kernel<<<blocks, threads>>>: launch GPU computation
  -> cudaDeviceSynchronize: wait and check execution errors
  -> cudaMemcpy: copy results GPU -> CPU
  -> release GPU buffers and return
```

`qkv_project()` runs on the CPU. `__global__ qkv_kernel()` runs on the GPU.
Each GPU thread computes one output element; a 16-by-16 thread block covers
part of the token/output grid. Boundary checks cover incomplete edge tiles.
The test includes a 17-by-19 output to exercise those edges.

The kernel is deliberately simple: it does not use shared-memory tiling,
Tensor Cores, cuBLAS, or optimized fusion. It is for inspecting the call chain,
not measuring production performance. Allocating and transferring weights on
every call is also a teaching choice; an engine would usually keep weights and
KV buffers resident and use streams/events to manage execution.

## Relationship to vLLM

| This POC | vLLM |
|---|---|
| `demo.py` orchestrates projection | Model runner orchestrates model execution |
| `ctypes` loads a C ABI library | PyTorch extensions register native operators |
| `project()` calls a native symbol | Layers call PyTorch ops or `torch.ops` |
| Explicit host float buffers | PyTorch tensors with device/layout metadata |
| CPU loop or simple CUDA kernel | Optimized PyTorch, CUDA, Triton, and external kernels |

This demonstrates the foreign-function boundary using a small, inspectable
interface. It does not implement vLLM's PyTorch dispatcher integration.
Default unquantized QKV projections in vLLM commonly enter PyTorch through
`torch.nn.functional.linear`; specialized implementations have other paths.

There is no tokenizer, trained model, attention operation, positional encoding,
KV cache, or text generation in the original QKV demo. Inputs are two manually chosen
embedding-like vectors and weights are illustrative. Q, K, and V are only the
projection results; rotary encoding and attention would follow in a model.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing compiler | On macOS, install Apple's command-line developer tools with `xcode-select --install`; on Linux, install `clang++` or `g++` |
| Missing native library | Run `python3 build.py --backend cpu` or `--backend metal` before the relevant demo/tests |
| Metal API build errors | The current Metal host code requires macOS 15+ and a compatible SDK/compiler |
| No Metal GPU available | Use a Metal-capable Mac; restricted sandbox/remote sessions may not expose the GPU |
| GPU tests are skipped | Set `ATTENTION_TEST_GPU=1` when running the test suite |
| Python trial takes a long time | Reduce prefix, hidden size, width, steps, or `--python-repeats` |
| Numerical mismatch | The run fails rather than publishing a speedup; check dimensions, kernel changes, and precision |
| CUDA compiler unavailable | CUDA projection requires Linux, `nvcc`, and a compatible NVIDIA GPU; it does not execute on Apple Silicon |

CPU/reference tests can run without Metal access after building the CPU library:

```bash
python3 build.py --backend cpu
python3 -m unittest -v
```

The complete GPU suite additionally requires the Metal library and device:

```bash
python3 build.py --backend metal
ATTENTION_TEST_GPU=1 python3 -m unittest -v
```

Tests cover numerical results, invalid buffers/dimensions, one-token attention,
non-square shapes, and cache resets between GPU modes. NVIDIA CUDA remains an
optional projection example and has not been validated on this Mac.
