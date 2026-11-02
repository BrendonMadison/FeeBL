"""In-memory tests of the paired uncertainty and native reader contracts."""

import copy
import unittest

import numpy as np

from paired_spectra import (
    COLUMNS, mean_covariance, paired_comparison, read_spectrum,
    remap_mass, timing_record, validate_inventory,
)


def spectrum(rows, seed=17):
    return "\n".join([
        "#FEEBL-NATIVE 1.0", f"#rndm_seed {seed}",
        "#BLOCK pair_spectrum", "#COLUMNS " + COLUMNS, *rows,
        "#underflow 0", "#overflow 0", "#mean_W 2 +- 0.1", "#rms_W 0.3",
        f"#END pair_spectrum records {len(rows)}",
    ])


class PairedStatistics(unittest.TestCase):
    def test_shared_noise_is_subtracted_once(self):
        noise = np.array([-3., -1., 1., 3.])
        delta = np.array([0., 2., 1., 3.])
        cpu = (100 + 10 * noise)[:, None]
        gpu = cpu + delta[:, None]
        result = paired_comparison(cpu, gpu)
        expected = np.var(delta, ddof=1) / 4
        self.assertAlmostEqual(result["difference_covariance_of_mean"][0][0], expected)
        self.assertGreater(result["cpu_covariance_of_mean"][0][0], 100 * expected)
        self.assertAlmostEqual(result["pull"][0], np.mean(delta) / np.sqrt(expected))
        self.assertEqual(result["pull_error"], [1.0])

    def test_negative_correlation_increases_combined_error(self):
        cpu = np.array([[1.], [2.], [4.], [5.]])
        gpu = 10 - cpu
        result = paired_comparison(cpu, gpu)
        self.assertAlmostEqual(result["difference_covariance_of_mean"][0][0],
                               4 * result["cpu_covariance_of_mean"][0][0])

    def test_identical_pairs_have_explicit_zero_convention(self):
        cpu = np.array([[0., 1.], [0., 2.], [0., 3.], [0., 4.]])
        result = paired_comparison(cpu, cpu)
        self.assertEqual(result["pull_state"], ["zero_denominator_identity_convention"] * 2)
        self.assertEqual(result["pull"], [0., 0.])
        self.assertEqual(result["pull_error"], [1., 1.])
        np.testing.assert_array_equal(result["combined_sem"], [0., 0.])

    def test_constant_nonzero_difference_is_explicit(self):
        cpu = np.arange(4.)[:, None]
        result = paired_comparison(cpu, cpu + 2)
        self.assertEqual(result["pull_state"], ["nonzero_difference_without_seed_scatter"])
        self.assertEqual(result["difference_gpu_minus_cpu"], [2.0])
        self.assertEqual(result["pull"], ["Infinity"])
        self.assertEqual(paired_comparison(cpu + 2, cpu)["pull"], ["-Infinity"])
        self.assertEqual(result["pull_error"], [1.0])

    def test_rebinning_commutes_with_full_covariance(self):
        samples = np.array([[1., 2., 3.], [2., 0., 4.], [3., 4., 2.], [0., 1., 5.]])
        transform = np.array([[1., 1., 0.], [0., 0., 1.]])
        _, fine = mean_covariance(samples)
        _, coarse = mean_covariance(samples @ transform.T)
        np.testing.assert_allclose(coarse, transform @ fine @ transform.T, atol=1e-15)

    def test_sparse_reader_and_mass_conservation(self):
        text = spectrum(["0 1 3 3 0 0 0 0 0 0 1", "2 3 5 5 0 0 0 0 0 0 1"])
        bins, metadata = read_spectrum(text, 17)
        np.testing.assert_array_equal(remap_mass(bins, [0, 1, 2, 3]), [3, 0, 5])
        np.testing.assert_array_equal(remap_mass(bins, [0, 3]), [8])
        self.assertEqual(metadata["rms_W"], .3)

    def test_reader_rejects_wrong_seed_overlap_and_truncation(self):
        text = spectrum(["0 1 3 3 0 0 0 0 0 0 1"])
        for corrupt in [text.replace("seed 17", "seed 19"),
                        text.replace("records 1", "records 2"),
                        spectrum(["0 2 3 3 0 0 0 0 0 0 1", "1 3 5 5 0 0 0 0 0 0 1"])]:
            with self.assertRaises(ValueError):
                read_spectrum(corrupt, 17)

    def test_inventory_rejects_duplicate_or_missing_seed(self):
        document = {"schema": "feebl.gpu-cpu-stress-analysis/1", "seeds": [1, 2, 3, 4],
                    "variants": ["cpu", "gpu"],
                    "comparisons": [{"id": "gpu_cpu", "cpu": "cpu", "gpu": "gpu"}],
                    "runs": [{"variant": v, "seed": s, "physics_input_sha256": str(s),
                              "source_identity": "source", "binary_sha256": v}
                             for v in ["cpu", "gpu"] for s in [1, 2, 3, 4]]}
        validate_inventory(document)
        for change in ("duplicate", "missing", "physics", "binary"):
            bad = copy.deepcopy(document)
            if change == "duplicate": bad["runs"].append(bad["runs"][0])
            if change == "missing": bad["runs"].pop()
            if change == "physics": bad["runs"][0]["physics_input_sha256"] = "different"
            if change == "binary": bad["runs"][0]["binary_sha256"] = "different"
            with self.assertRaises(ValueError): validate_inventory(bad)

    def test_runtime_quantities_are_not_conflated(self):
        row = {"variant": "gpu", "seed": 1, "binary_sha256": "x", "gpu_count": 1,
               "gpu_uuid": "GPU-example", "cpu_threads": 4}
        receipt = {"status": "EXECUTED", "returncode": 0, "seed": 1,
                   "binary_sha256": "x", "elapsed_seconds": 10,
                   "resources": {"user_cpu_seconds": 3, "system_cpu_seconds": 1}}
        stderr = " FEEBL GPU2 primary summary: device_work_calls=4 kernel_ms=500 h2d_ms=10 d2h_ms=20 setup_ms=30 total_ms=600"
        value = timing_record(row, receipt, stderr)
        self.assertEqual(value["wall_seconds"], 10)
        self.assertEqual(value["cpu_core_seconds"], 4)
        self.assertEqual(value["cpu_slot_seconds_reserved_for_process"], 40)
        self.assertEqual(value["gpu_device_seconds_reserved_for_process"], 10)
        self.assertEqual(value["cuda_event_kernel_seconds"], .5)
        self.assertEqual(value["gpu_solver_host_elapsed_seconds"], .6)

    def test_gpu4_boundary_time_is_not_kernel_or_solver_total(self):
        row = {"variant": "gpu", "seed": 1, "binary_sha256": "x", "gpu_count": 1,
               "gpu_uuid": "GPU-example", "cpu_threads": 16}
        receipt = {"status": "EXECUTED", "returncode": 0, "seed": 1,
                   "binary_sha256": "x", "elapsed_seconds": 10,
                   "resources": {"user_cpu_seconds": 3, "system_cpu_seconds": 1}}
        stderr = (" FEEBL GPU mode4 summary: implementation=encounter-dag-wave-batched-planmany-fields-v1 "
                  "batch_calls=4 kernel_launches=20 cufft_executes=8 kernel_ms=500 h2d_ms=10 "
                  "d2h_ms=20 setup_ms=30 host_pack_ms=40 host_unpack_ms=50 boundary_ms=700")
        value = timing_record(row, receipt, stderr)
        self.assertEqual(value["cuda_event_kernel_seconds"], .5)
        self.assertEqual(value["gpu_batch_boundary_host_elapsed_seconds"], .7)
        self.assertIsNone(value["gpu_solver_host_elapsed_seconds"])
        with self.assertRaises(ValueError):
            timing_record(row, receipt, stderr.replace("batch_calls=4", "batch_calls=0"))


if __name__ == "__main__":
    unittest.main()
