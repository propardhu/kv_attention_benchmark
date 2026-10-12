"""Benchmark Python CPU attention against handwritten Metal GPU attention with/without KV reuse."""

import argparse
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from attention_workload import MetalAttentionBackend, make_inputs, python_attention_reference


def summary(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values) * 1000,
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1] * 1000,
    }


def max_error(expected, actual):
    if len(expected) != len(actual) or any(len(a) != len(b) for a, b in zip(expected, actual)):
        raise RuntimeError("GPU output shape differs from reference")
    if any(not math.isfinite(value) for row in actual for value in row):
        raise RuntimeError("GPU output contains a nonfinite value")
    return max(abs(a - b) for row_a, row_b in zip(expected, actual)
               for a, b in zip(row_a, row_b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=int, default=256)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--python-repeats", type=int, default=3)
    parser.add_argument("--gpu-repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if min(args.prefix, args.steps, args.hidden, args.width,
           args.python_repeats, args.gpu_repeats) <= 0 or args.warmup < 1:
        parser.error("dimensions/repeats must be positive; warmup must be at least 1")
    dims = (args.prefix, args.steps, args.hidden, args.width)
    inputs, weights = make_inputs(*dims)
    setup_start = time.perf_counter()
    engine = MetalAttentionBackend(inputs, weights, *dims)
    setup_ms = (time.perf_counter() - setup_start) * 1000
    try:
        print(f"GPU: {engine.device}; dimensions: {dims}; setup: {setup_ms:.2f} ms", flush=True)
        for _ in range(args.warmup):
            engine.run(False)
            engine.run(True)
        python_samples, python_prefill, python_decode = [], [], []
        reference = None
        for repeat in range(args.python_repeats):
            start = time.perf_counter()
            reference, times = python_attention_reference(inputs, weights, *dims)
            python_samples.append(time.perf_counter() - start)
            python_prefill.append(times[0])
            python_decode.append(sum(times[1:]))
            print(f"Python repeat {repeat + 1}: {python_samples[-1] * 1000:.2f} ms", flush=True)

        data = {
            "python_cpu_uncached": {"total": python_samples, "prefill": python_prefill,
                           "decode": python_decode, "error": 0.0},
            "metal_gpu_uncached": {"total": [], "prefill": [], "decode": [],
                             "gpu_total": [], "gpu_decode": [], "error": 0.0},
            "metal_gpu_kv_cached": {"total": [], "prefill": [], "decode": [],
                             "gpu_total": [], "gpu_decode": [], "error": 0.0},
        }
        # Alternate order so one mode does not always get the earlier thermal state.
        for repeat in range(args.gpu_repeats):
            for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
                name = "metal_gpu_kv_cached" if cached else "metal_gpu_uncached"
                start = time.perf_counter()
                outputs, wall, gpu = engine.run(cached)
                duration = time.perf_counter() - start
                if any(not math.isfinite(t) or t <= 0 for t in wall + gpu):
                    raise RuntimeError("Backend did not provide positive finite timing samples")
                error = max_error(reference, outputs)
                if not math.isfinite(error) or error > 1e-4:
                    raise RuntimeError(f"{name} failed correctness check: max error {error}")
                item = data[name]
                item["total"].append(duration)
                item["prefill"].append(wall[0])
                item["decode"].append(sum(wall[1:]))
                item["gpu_total"].append(sum(gpu))
                item["gpu_decode"].append(sum(gpu[1:]))
                item["error"] = max(item["error"], error)
        metrics = {
            name: {**{key: summary(values) for key, values in item.items() if key != "error"},
                   "max_absolute_error": item["error"]}
            for name, item in data.items()
        }
        full_rows = sum(args.prefix + step for step in range(args.steps + 1))
        cached_rows = args.prefix + args.steps
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "python": platform.python_version(),
            "device": engine.device, "parameters": {k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()},
            "setup_ms_excluded": setup_ms,
            "work": {"uncached_kv_projected_rows": full_rows,
                     "cached_kv_projected_rows": cached_rows,
                     "kv_projection_work_ratio": full_rows / cached_rows,
                     "kv_cache_bytes": 2 * cached_rows * args.width * 4},
            "implementation": {
                "kernels": "Handwritten Metal Shading Language projection, score, and attention kernels",
                "native_host": "Handwritten Objective-C++ using the Apple Metal API",
                "python_packages": "Standard library only; no third-party Python packages",
                "system_dependencies": "Python, C++/Objective-C++ compiler, macOS Metal/Foundation frameworks and GPU driver",
                "scope": "Single-head attention; not a standalone GEMM or a full inference engine",
            },
            "metrics": metrics, "raw_seconds": data,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "kv_attention_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        gpu_full = metrics["metal_gpu_uncached"]["decode"]["median_ms"]
        gpu_cached = metrics["metal_gpu_kv_cached"]["decode"]["median_ms"]
        kernel_full = metrics["metal_gpu_uncached"]["gpu_decode"]["median_ms"]
        kernel_cached = metrics["metal_gpu_kv_cached"]["gpu_decode"]["median_ms"]
        lines = [
            "# Python CPU vs Handwritten Metal GPU: KV Attention Benchmark", "",
            f"Device: **{engine.device}**. Python {platform.python_version()}.", "",
            f"Workload: prefix {args.prefix}, {args.steps} decode steps, hidden size "
            f"{args.hidden}, attention width {args.width}; single head, batch 1, float32 GPU.", "",
            "| Path | Total median (ms) | Prefill median (ms) | Decode median (ms) | Total min / p95 (ms) | Max output error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, label in (("python_cpu_uncached", "1. Python CPU attention, uncached"),
                            ("metal_gpu_uncached", "2. Handwritten Metal GPU attention, uncached"),
                            ("metal_gpu_kv_cached", "3. Handwritten Metal GPU attention, KV cached")):
            item = metrics[name]
            lines.append(f"| {label} | {item['total']['median_ms']:.3f} | "
                         f"{item['prefill']['median_ms']:.3f} | {item['decode']['median_ms']:.3f} | "
                         f"{item['total']['min_ms']:.3f} / {item['total']['p95_ms']:.3f} | "
                         f"{item['max_absolute_error']:.3g} |")
        lines.extend([
            "", f"GPU KV-cache decode speedup, completed-call timing: **{gpu_full / gpu_cached:.2f}x**.",
            f"GPU execution-only decode: {kernel_full:.3f} ms without cache vs "
            f"{kernel_cached:.3f} ms with cache (**{kernel_full / kernel_cached:.2f}x**).", "",
            f"KV projection rows: {full_rows} without cache vs {cached_rows} with cache "
            f"({full_rows / cached_rows:.2f}x less KV projection work). "
            f"Persistent KV capacity: {report['work']['kv_cache_bytes']:,} bytes.", "",
            "## Measurement Boundaries", "",
            f"- Python repeats: {args.python_repeats}; GPU repeats: {args.gpu_repeats}; "
            f"GPU warmup runs per mode: {args.warmup}.",
            f"- Native initialization/compilation/buffer upload took {setup_ms:.3f} ms and is excluded.",
            "- Total times include the call, completed execution, and result conversion. "
            "Inputs and weights are already prepared and GPU-resident for both GPU modes.",
            "- Prefill/decode times measure per-step work; GPU times include encoding, "
            "submission, and waiting. GPU execution-only times use Metal command-buffer timestamps.",
            "- Each GPU step waits for completion. Both modes use identical kernels, "
            "precision, buffers, submission counts, and attention calculations.",
            "- Every GPU trial is checked against every prefill/decode reference output "
            "with maximum absolute error <= 1e-4. Python arithmetic uses doubles on float32 inputs.",
            "- Pure Python uses explicit loops, not NumPy/BLAS/PyTorch. Its comparison "
            "with GPU includes interpreter and algorithm-implementation differences; "
            "it does not isolate foreign-function-call overhead or compare optimized CPU BLAS.",
            "- All projection, score, softmax, and weighted-value calculations are handwritten. "
            "The GPU kernels use Metal Shading Language; the host bridge uses Objective-C++.",
            "- No third-party Python packages or numerical libraries are required. "
            "System dependencies are Python, a compiler, Apple's Metal/Foundation frameworks, and the GPU driver.",
            "- This is a small single-layer attention benchmark with predetermined token "
            "vectors, not a trained autoregressive model or a vLLM throughput benchmark.",
            "- Cache reuse saves historical K/V projections. Attention still reads the "
            "whole retained prefix. Prefill and GPU submission overhead remain.",
            "- A speedup below 1 means slower for this measurement. Results depend on "
            "shape, thermal state, system activity, and GPU dispatch overhead.",
            "- JSON contains all timing samples and parameters. Re-run to assess stability.", "",
        ])
        report_path = args.output / "KV_ATTENTION_BENCHMARK.md"
        report_path.write_text("\n".join(lines))
        print("\n".join(lines[:14]), flush=True)
        print(f"Report: {report_path.resolve()}", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
