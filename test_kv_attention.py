import os
import unittest

from attention_workload import MetalAttentionBackend, make_inputs, python_attention_reference
from benchmark_kv_attention import max_error


class ReferenceTests(unittest.TestCase):
    def test_kv_reuse_matches_recomputation_at_every_step(self):
        dims = (5, 4, 7, 3)
        inputs, weights = make_inputs(*dims)
        uncached, _ = python_attention_reference(inputs, weights, *dims)
        cached, _ = python_attention_reference(inputs, weights, *dims, cached=True)
        self.assertEqual(uncached, cached)

    def test_one_token_attention_returns_its_value_projection(self):
        result, _ = python_attention_reference([2, 3], [1, 0, 0, 1, 1, 1], 1, 0, 2, 1)
        self.assertEqual(result, [[5]])

    def test_correctness_check_rejects_missing_or_nonfinite_output(self):
        for actual in ([], [[float("nan")]], [[float("inf")]], [[1, 2]]):
            with self.subTest(actual=actual), self.assertRaises(RuntimeError):
                max_error([[1]], actual)


@unittest.skipUnless(os.environ.get("ATTENTION_TEST_GPU") == "1", "Opt-in Metal GPU tests")
class GPUTests(unittest.TestCase):
    def test_modes_match_reference_with_reset_and_non_square_dimensions(self):
        for dims in ((1, 0, 1, 1), (5, 4, 7, 3), (17, 3, 9, 19)):
            with self.subTest(dims=dims):
                inputs, weights = make_inputs(*dims)
                expected, _ = python_attention_reference(inputs, weights, *dims)
                with MetalAttentionBackend(inputs, weights, *dims) as engine:
                    for cached in (True, False, True):
                        actual, wall, gpu = engine.run(cached)
                        self.assertEqual(len(actual), len(expected))
                        self.assertTrue(all(t > 0 for t in wall))
                        self.assertTrue(all(t > 0 for t in gpu))
                        for row_a, row_b in zip(actual, expected):
                            for a, b in zip(row_a, row_b):
                                self.assertAlmostEqual(a, b, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
