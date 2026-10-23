"""Repeatable checks for the shared CCF RV calculation.
The frozen synthetic reference is in fixtures/exsplosyn_reference.npz.
"""

import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import rv_ccf


class RVTests(unittest.TestCase):
    # unittest runs methods named test_*; the class name RVTests is our choice.
    @classmethod
    def setUpClass(cls):
        """Load shared inputs and frozen original EXSPLOSYN results once.

        unittest calls this before running the test methods in this class."""
        # Reference provenance: generated with the untouched EXSPLOSYN code at
        # commit 2546db926d261d6e2512f63f65847b3ea65e9c24, using a spectral
        # step of 0.306 km/s, oversampling factor 50 and RV half-range 40 km/s.
        # Require exact RV, coarse/fine grid, CCF and wavelength-bound agreement.
        # Injected-shift checks elsewhere are sanity checks, not RV precision claims.
        with np.load(
            Path(__file__).parent / "fixtures/exsplosyn_reference.npz"
        ) as data:
            cls.ref = {key: data[key] for key in data.files}

    def inputs(self):
        """Return the four input arrays; this helper is not itself a test."""
        r = self.ref
        return r["wavelength"], r["flux"], r["template_wavelength"], r["template_flux"]

    def test_exact_original_rvs_profiles_and_interval(self):
        """Reproduce the frozen original EXSPLOSYN outputs exactly.

        Compare RVs, wavelength bounds, and every coarse/fine grid and CCF value.
        This protects extraction compatibility, not absolute physical accuracy."""
        result = rv_ccf.measure_rv(
            *self.inputs(), spectrum_vel_step=0.306, return_ccf=True
        )
        np.testing.assert_array_equal(result.rv, self.ref["expected_rv"])
        self.assertEqual(
            [result.report["wavelength_start"], result.report["wavelength_stop"]],
            self.ref["expected_slice"].tolist(),
        )
        for i, profile in enumerate(result.profiles):
            for key in ["coarse_grid", "coarse_ccf", "fine_grid", "fine_ccf"]:
                np.testing.assert_array_equal(profile[key], self.ref[f"{key}_{i}"])

    def test_inferred_log_sampling_and_single_input(self):
        """Infer sampling from an equal-log-wavelength grid for one spectrum.

        Check the fine step, scalar RV, and absence of unrequested profiles.
        The injected -2.37 km/s shift is checked within 0.05 km/s (50 m/s)."""
        r = self.ref
        w = np.exp(np.linspace(np.log(9990), np.log(10010), 4001))
        flux = np.interp(w, r["wavelength"], r["flux"][0])
        result = rv_ccf.measure_rv(
            w, flux, r["template_wavelength"], r["template_flux"]
        )
        expected = 299792.458 * np.median(np.diff(np.log(w)))
        self.assertAlmostEqual(result.report["fine_step_kms"], expected / 50, places=14)
        self.assertIsInstance(result.rv, float)
        self.assertLess(abs(result.rv + 2.37), 0.05)
        self.assertIsNone(result.profiles)

    def test_irregular_grid_uses_original_samples(self):
        """Recover injected shifts after removing every third wavelength sample.

        Check RVs within 50 m/s and the reported count of retained pixels;
        the count alone does not prove that no internal resampling occurred."""
        w, f, tw, tf = self.inputs()
        keep = np.arange(w.size) % 3 != 0
        result = rv_ccf.measure_rv(w[keep], f[:, keep], tw, tf)
        np.testing.assert_allclose(result.rv, [-2.37, 0.0, 3.418], atol=0.05)
        self.assertEqual(result.report["usable_wavelength_points"], keep.sum())

    def test_retry_reruns_entire_batch_and_matches_original(self):
        """Expand +/-5 to +/-10 km/s to recover a +6 km/s shifted spectrum.

        Record both input rows on both attempts, then compare with a search
        started directly at +/-10 km/s using the extracted search routine."""
        w, _, tw, tf = self.inputs()
        flux = np.array(
            [np.interp(w / (1 + v / 299792.458), tw, tf) for v in [0.0, 6.0]]
        )
        seen = []
        result = rv_ccf.measure_rv(
            w,
            flux,
            tw,
            tf,
            rv_half_range=5,
            spectrum_vel_step=0.306,
            return_ccf=True,
            debug_callback=lambda **p: seen.append(
                (p["attempt_index"], p["phase_index"])
            ),
        )
        self.assertEqual(result.report["retries_used"], 1)
        self.assertEqual(result.report["half_range_kms"], 10)
        self.assertEqual(set(seen), {(0, 0), (0, 1), (1, 0), (1, 1)})
        direct, report = rv_ccf.coarse_to_fine_ccf_rv_search(
            w, flux, tw, tf, half_range=10, fine_step=0.00612
        )
        np.testing.assert_array_equal(result.rv, direct)
        self.assertEqual(result.report["wavelength_start"], report["wavelength_start"])

    def test_multiple_retries(self):
        """Recover a +12 km/s shift using two expansions: +/-5, 10, then 20.

        Check that both retries occur and the recovered RV is within 50 m/s."""
        w, _, tw, tf = self.inputs()
        flux = np.interp(w / (1 + 12 / 299792.458), tw, tf)
        result = rv_ccf.measure_rv(w, flux, tw, tf, rv_half_range=5, n_retries=2)
        self.assertEqual(result.report["retries_used"], 2)
        self.assertLess(abs(result.rv - 12), 0.05)

    def test_flat_and_exhausted_search_preserve_failure_policy(self):
        """Keep the expected NaN and error behavior for unsuccessful searches.

        A flat spectrum returns NaN without expansion. A +12 km/s shift is
        outside +/-5 and +/-10: no retry returns NaN; an exhausted retry raises."""
        w, _, tw, tf = self.inputs()
        flat = rv_ccf.measure_rv(w, np.ones_like(w), tw, tf)
        self.assertTrue(np.isnan(flat.rv))
        self.assertEqual(flat.report["failures"], ["flat_ccf"])
        shifted = np.interp(w / (1 + 12 / 299792.458), tw, tf)
        no_retry = rv_ccf.measure_rv(w, shifted, tw, tf, rv_half_range=5, n_retries=0)
        self.assertTrue(np.isnan(no_retry.rv))
        with self.assertRaisesRegex(ValueError, "after retrying"):
            rv_ccf.measure_rv(w, shifted, tw, tf, rv_half_range=5, n_retries=1)

    def test_coverage_reselected_on_retry(self):
        """Shrink the usable wavelength interval when the RV range doubles.

        Temporarily wrap the real interval-selection function to record its
        bounds without changing its calculation; patch restores it afterward."""
        w, f, _, _ = self.inputs()
        intervals = []
        # Exercise the actual interval selection while retaining realistic spectra.
        original = rv_ccf.find_common_wavelength_slice

        def record(*args):
            interval = original(*args)
            intervals.append(interval)
            return interval

        flux = np.interp(w / (1 + 6 / 299792.458), w, f[1])
        with patch.object(rv_ccf, "find_common_wavelength_slice", side_effect=record):
            result = rv_ccf.measure_rv(w, flux, w, f[1], rv_half_range=5)
        self.assertEqual(result.report["retries_used"], 1)
        self.assertGreater(intervals[1][0], intervals[0][0])
        self.assertLess(intervals[1][1], intervals[0][1])

    def test_sampling_and_coverage_bounds(self):
        """Reject invalid settings and a range with insufficient template coverage.

        Each subTest names one invalid setting. assertRaises requires the
        expected ValueError; raising that error is a passing outcome here."""
        args = self.inputs()
        for options in [
            dict(ccf_rv_oversample_factor=0.5),
            dict(rv_half_range=1),
            dict(spectrum_vel_step=50, ccf_rv_oversample_factor=50),
            dict(spectrum_vel_step=0),
            dict(n_retries=-1),
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                rv_ccf.measure_rv(*args, **options)
        with self.assertRaisesRegex(ValueError, "fewer than four"):
            rv_ccf.measure_rv(*args, rv_half_range=1000)

    def test_explicit_step_skips_inference_and_optional_profiles_do_not_change_rv(self):
        """Use supplied sampling and keep RVs identical with or without profiles.

        Check that inference was skipped and the fine step remains 0.306/50."""
        a = rv_ccf.measure_rv(*self.inputs(), spectrum_vel_step=0.306)
        b = rv_ccf.measure_rv(*self.inputs(), spectrum_vel_step=0.306, return_ccf=True)
        self.assertFalse(a.report["sampling_inferred"])
        self.assertEqual(a.report["fine_step_kms"], 0.306 / 50)
        np.testing.assert_array_equal(a.rv, b.rv)


# Start the unittest runner only when this file is executed directly.
if __name__ == "__main__":
    unittest.main()
