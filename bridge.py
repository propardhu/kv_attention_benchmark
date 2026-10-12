"""Declare the native ABI and keep Python-owned float buffers alive during calls."""

import ctypes
import operator

from build import library_path


class QKVProjector:
    def __init__(self, backend: str = "cpu") -> None:
        if backend not in ("cpu", "cuda"):
            raise ValueError("backend must be 'cpu' or 'cuda'")
        path = library_path(backend)
        if not path.exists():
            raise FileNotFoundError(
                f"Build the backend first: python3 build.py --backend {backend}"
            )
        self.library = ctypes.CDLL(str(path))
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.library.qkv_project.argtypes = [
            float_pointer, float_pointer, float_pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.library.qkv_project.restype = ctypes.c_int
        self.library.qkv_backend.argtypes = []
        self.library.qkv_backend.restype = ctypes.c_char_p
        self.library.qkv_error_message.argtypes = [ctypes.c_int]
        self.library.qkv_error_message.restype = ctypes.c_char_p

    @property
    def backend(self) -> str:
        return self.library.qkv_backend().decode("ascii")

    def project(self, inputs, weights, *, tokens: int, hidden: int, outputs: int):
        dimensions = []
        for value in (tokens, hidden, outputs):
            try:
                dimension = operator.index(value)
            except TypeError as error:
                raise ValueError("dimensions must be positive integers") from error
            if isinstance(value, bool) or not 0 < dimension <= 2**31 - 1:
                raise ValueError("dimensions must fit a positive signed 32-bit integer")
            dimensions.append(dimension)
        tokens, hidden, outputs = dimensions
        if len(inputs) != tokens * hidden:
            raise ValueError("input length must equal tokens * hidden")
        if len(weights) != outputs * hidden:
            raise ValueError("weight length must equal outputs * hidden")

        input_buffer = (ctypes.c_float * len(inputs))(*inputs)
        weight_buffer = (ctypes.c_float * len(weights))(*weights)
        output_buffer = (ctypes.c_float * (tokens * outputs))()
        status = self.library.qkv_project(
            input_buffer, weight_buffer, output_buffer, tokens, hidden, outputs
        )
        if status != 0:
            message = self.library.qkv_error_message(status).decode("utf-8")
            raise RuntimeError(f"Native projection failed ({status}): {message}")
        return [
            list(output_buffer[token * outputs:(token + 1) * outputs])
            for token in range(tokens)
        ]


def split_qkv(rows, q_size: int, k_size: int, v_size: int):
    sizes = (q_size, k_size, v_size)
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0
           for size in sizes):
        raise ValueError("Q, K, V sizes must be positive integers")
    if any(len(row) != sum(sizes) for row in rows):
        raise ValueError("each row must contain q_size + k_size + v_size values")
    return (
        [row[:q_size] for row in rows],
        [row[q_size:q_size + k_size] for row in rows],
        [row[q_size + k_size:] for row in rows],
    )
