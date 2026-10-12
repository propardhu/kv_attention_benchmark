"""Handwritten Python CPU attention and a ctypes bridge to our Metal kernels."""

import ctypes
import math
import random
import time

from build import library_path


def make_inputs(prefix, steps, hidden, width, seed=42):
    rng = random.Random(seed)
    float32 = lambda x: ctypes.c_float(x).value
    inputs = [float32(rng.uniform(-1, 1)) for _ in range((prefix + steps) * hidden)]
    scale = 1 / math.sqrt(hidden)
    weights = [float32(rng.uniform(-scale, scale)) for _ in range(3 * width * hidden)]
    return inputs, weights


def python_attention_reference(inputs, weights, prefix, steps, hidden, width, cached=False):
    """Compute the last token's attention at prefill and every decode step."""
    rows, results, timings = [], [], []
    for step in range(steps + 1):
        start = time.perf_counter()
        length = prefix + step
        first = len(rows) if cached else 0
        if not cached:
            rows = []
        for token in range(first, length):
            rows.append([
                sum(inputs[token * hidden + k] * weights[column * hidden + k]
                    for k in range(hidden))
                for column in range(width, 3 * width)
            ])
        query = [sum(inputs[(length - 1) * hidden + k] * weights[d * hidden + k]
                     for k in range(hidden)) for d in range(width)]
        scores = [sum(query[d] * row[d] for d in range(width)) /
                  math.sqrt(width) for row in rows]
        maximum = max(scores)
        probabilities = [math.exp(score - maximum) for score in scores]
        denominator = sum(probabilities)
        result = [sum(probabilities[t] * rows[t][width + d]
                      for t in range(length)) / denominator for d in range(width)]
        timings.append(time.perf_counter() - start)
        results.append(result)
    return results, timings


class MetalAttentionBackend:
    def __init__(self, inputs, weights, prefix, steps, hidden, width):
        self.handle = None
        dims = (prefix, steps, hidden, width)
        if any(not isinstance(x, int) or isinstance(x, bool) for x in dims):
            raise ValueError("dimensions must be integers")
        if prefix <= 0 or steps < 0 or hidden <= 0 or width <= 0:
            raise ValueError("prefix/hidden/width must be positive and steps nonnegative")
        if max(prefix + steps, hidden, 3 * width) > 65535:
            raise ValueError("this teaching backend limits dimensions to 65535")
        if len(inputs) != (prefix + steps) * hidden or len(weights) != 3 * width * hidden:
            raise ValueError("input or weight buffer length does not match dimensions")
        path = library_path("metal")
        if not path.exists():
            raise FileNotFoundError("Build the GPU backend: python3 build.py --backend metal")
        self.lib = ctypes.CDLL(str(path))
        floats = ctypes.POINTER(ctypes.c_float)
        doubles = ctypes.POINTER(ctypes.c_double)
        self.lib.attention_create.argtypes = [floats, floats] + [ctypes.c_int] * 4
        self.lib.attention_create.restype = ctypes.c_void_p
        self.lib.attention_run.argtypes = [ctypes.c_void_p, ctypes.c_int, doubles, doubles]
        self.lib.attention_run.restype = ctypes.c_int
        self.lib.attention_output.argtypes = [ctypes.c_void_p]
        self.lib.attention_output.restype = floats
        self.lib.attention_device.argtypes = [ctypes.c_void_p]
        self.lib.attention_device.restype = ctypes.c_char_p
        self.lib.attention_destroy.argtypes = [ctypes.c_void_p]
        self.lib.attention_destroy.restype = None
        self.lib.attention_error.argtypes = []
        self.lib.attention_error.restype = ctypes.c_char_p
        x = (ctypes.c_float * len(inputs))(*inputs)
        w = (ctypes.c_float * len(weights))(*weights)
        self.handle = self.lib.attention_create(x, w, *dims)
        if not self.handle:
            raise RuntimeError(self.lib.attention_error().decode())
        self.steps, self.width = steps, width
        self.device = self.lib.attention_device(self.handle).decode()

    def run(self, cached):
        if not self.handle:
            raise RuntimeError("GPU context is closed")
        wall = (ctypes.c_double * (self.steps + 1))()
        gpu = (ctypes.c_double * (self.steps + 1))()
        if self.lib.attention_run(self.handle, int(cached), wall, gpu):
            raise RuntimeError(self.lib.attention_error().decode())
        output = self.lib.attention_output(self.handle)
        rows = [[output[t * self.width + d] for d in range(self.width)]
                for t in range(self.steps + 1)]
        return rows, list(wall), list(gpu)

    def close(self):
        if self.handle:
            self.lib.attention_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
