"""Project two token vectors with combined QKV weights, then split the result."""

import argparse

from bridge import QKVProjector, split_qkv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    projector = QKVProjector(args.backend)

    # Two already-embedded tokens, each with hidden size 3.
    inputs = [1, 2, 3, 4, 5, 6]
    # Six projection rows: two Q rows, two K rows, then two V rows.
    weights = [
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 0,
        0, 1, 1,
        1, 0, 1,
    ]

    print("Python: owns input [2, 3] and weights [6, 3] float buffers")
    print("Bridge: ctypes -> shared library ->", projector.backend)
    result = projector.project(inputs, weights, tokens=2, hidden=3, outputs=6)
    q, k, v = split_qkv(result, q_size=2, k_size=2, v_size=2)
    print("Native result X @ W.T:", result)
    print("Python splits the combined result:")
    print("Q:", q)
    print("K:", k)
    print("V:", v)


if __name__ == "__main__":
    main()
