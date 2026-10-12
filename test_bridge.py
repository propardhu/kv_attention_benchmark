import ctypes
import os
import random
import unittest

from bridge import QKVProjector, split_qkv


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.projector = QKVProjector(os.environ.get("QKV_TEST_BACKEND", "cpu"))

    def test_projection_matches_reference_for_non_square_shapes(self):
        rng = random.Random(19)
        for tokens, hidden, outputs in ((1, 1, 1), (2, 3, 6), (3, 7, 5), (17, 9, 19)):
            with self.subTest(shape=(tokens, hidden, outputs)):
                inputs = [ctypes.c_float(rng.uniform(-2, 2)).value
                          for _ in range(tokens * hidden)]
                weights = [ctypes.c_float(rng.uniform(-2, 2)).value
                           for _ in range(outputs * hidden)]
                result = self.projector.project(
                    inputs, weights, tokens=tokens, hidden=hidden, outputs=outputs
                )
                for token in range(tokens):
                    for column in range(outputs):
                        expected = sum(
                            inputs[token * hidden + k] * weights[column * hidden + k]
                            for k in range(hidden)
                        )
                        self.assertAlmostEqual(result[token][column], expected, delta=1e-5)

    def test_combined_projection_splits_into_q_k_v(self):
        rows = self.projector.project(
            [1, 2, 3],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
            tokens=1, hidden=3, outputs=6,
        )
        self.assertEqual(split_qkv(rows, 2, 2, 2), ([[1, 2]], [[3, 3]], [[5, 4]]))

    def test_python_rejects_mismatched_buffers(self):
        with self.assertRaisesRegex(ValueError, "input length"):
            self.projector.project([1], [1, 2], tokens=1, hidden=2, outputs=1)
        with self.assertRaisesRegex(ValueError, "weight length"):
            self.projector.project([1, 2], [1], tokens=1, hidden=2, outputs=1)

    def test_python_rejects_invalid_dimensions_before_native_call(self):
        for dimension in (0, -1, 1.5, True, 2**31):
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                self.projector.project([1], [1], tokens=dimension, hidden=1, outputs=1)

    def test_native_abi_rejects_null_pointers_and_invalid_dimensions(self):
        self.assertEqual(self.projector.library.qkv_project(None, None, None, 1, 1, 1), -1)
        buffer = (ctypes.c_float * 1)(1)
        self.assertEqual(
            self.projector.library.qkv_project(buffer, buffer, buffer, 0, 1, 1), -1
        )

    def test_split_rejects_invalid_width(self):
        with self.assertRaises(ValueError):
            split_qkv([[1, 2, 3]], 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
