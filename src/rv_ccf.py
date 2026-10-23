"""Doppler CCF radial velocities for prepared target and template spectra.

Numerical correlation and peak search extracted from EXSPLOSYN 0.9.1.
Wavelengths are in Angstroms and velocities in km/s. Positive RV redshifts
the template to match the target. Inputs must already be continuum flattened.
"""

from dataclasses import dataclass
import logging

import numpy as np
from numba import njit
from scipy.signal import find_peaks

__version__ = "0.1.0"
__all__ = ["measure_rv", "RVResult"]
logger = logging.getLogger(__name__)


@njit(cache=True)
def _calculate_doppler_ccfs(
    rv_grid,
    wavelength,
    phase_spectra,
    template_wavelength,
    template_flux,
    c_kms,
    weights,
):
    """Correlate Doppler-shifted template features with each phase spectrum.

    Subtract each spectrum's mean and sum its product with the mean-subtracted
    shifted template. This measures spectral alignment at each trial RV.
    Interpolate the template once per RV and reuse it across phases, keeping
    only the current shifted template in memory. Return (n_phases, n_rvs) CCFs.
    """
    n_phases = phase_spectra.shape[0]
    centered_spectra = np.empty_like(phase_spectra)
    for phase_index in range(n_phases):
        centered_spectra[phase_index] = phase_spectra[phase_index] - np.mean(
            phase_spectra[phase_index]
        )

    ccf_curves = np.zeros((n_phases, rv_grid.size), dtype=np.float64)
    for rv_index in range(rv_grid.size):
        # Stream one RV at a time; do not retain a full shifted-template bank.
        # lambda_shifted = lambda_template * (1 + RV/c); positive RV is a redshift.
        shifted_wavelength = template_wavelength * (1.0 + rv_grid[rv_index] / c_kms)
        shifted_template = np.interp(wavelength, shifted_wavelength, template_flux)
        shifted_template *= weights
        shifted_template -= np.mean(shifted_template)

        for phase_index in range(n_phases):
            # Sum the spectral products in wavelength order for each phase.
            correlation = 0.0
            for wavelength_index in range(wavelength.size):
                correlation += (
                    centered_spectra[phase_index, wavelength_index]
                    * shifted_template[wavelength_index]
                )
            ccf_curves[phase_index, rv_index] = correlation
    return ccf_curves


def calculate_ccf_batch(
    wavelength,
    phase_spectra,
    template_wavelength,
    template_flux,
    rv_min,
    rv_max,
    rv_step,
    edge_pixels=0,
):
    """Compute Doppler CCFs for phases sharing one wavelength grid.

    Shift the template at each trial RV and compare its spectral features with
    all phases using mean-subtracted dot products. Template interpolation is
    shared across phases; calculation runs in a single process with unit weights.

    Args:
        wavelength: Common phase-spectrum wavelength grid in Angstroms.
        phase_spectra: Flux array shaped (n_phases, n_wavelengths).
        template_wavelength, template_flux: Template wavelength and flux arrays.
        rv_min, rv_max, rv_step: Trial RV limits and spacing in km/s;
            np.arange includes rv_min and excludes rv_max.
        edge_pixels: Number of samples removed from each spectrum edge (default 0).

    Returns:
        Float64 CCF array shaped (n_phases, n_rvs), or None if the shifted
        template cannot cover the measurement interval at every trial RV.

    Raises:
        ValueError: Input array sizes or dimensions are invalid, or edge trimming
            removes every spectrum sample.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    phase_spectra = np.asarray(phase_spectra, dtype=np.float64)
    template_wavelength = np.asarray(template_wavelength, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError(
            "Spectrum wavelength must be a nonempty one-dimensional array."
        )
    if (
        template_wavelength.ndim != 1
        or template_flux.ndim != 1
        or template_wavelength.size == 0
        or template_wavelength.size != template_flux.size
    ):
        raise ValueError(
            "Template wavelength and flux must be nonempty one-dimensional arrays "
            "with matching sizes."
        )
    if phase_spectra.ndim != 2 or phase_spectra.shape[1] != wavelength.size:
        raise ValueError("Phase spectra must have shape (n_phases, n_wavelengths).")

    if edge_pixels > 0:
        wavelength = wavelength[edge_pixels:-edge_pixels]
        phase_spectra = phase_spectra[:, edge_pixels:-edge_pixels]
    if wavelength.size == 0:
        raise ValueError("Edge trimming leaves no usable spectrum wavelength samples.")
    # Keep each phase's wavelength samples contiguous after edge slicing.
    phase_spectra = np.ascontiguousarray(phase_spectra)
    weights = np.ones_like(wavelength, dtype=np.float64)
    c_kms = 299792.458
    # The template must cover the measurement interval at both RV limits.
    covered_max = template_wavelength[-1] * (1.0 + rv_min / c_kms)
    covered_min = template_wavelength[0] * (1.0 + rv_max / c_kms)
    if covered_min > wavelength[0] or covered_max < wavelength[-1]:
        return None
    rv_grid = np.arange(rv_min, rv_max, rv_step, dtype=np.float64)
    return _calculate_doppler_ccfs(
        rv_grid,
        wavelength,
        phase_spectra,
        template_wavelength,
        template_flux,
        c_kms,
        weights,
    )


def _refine_ccf_peak(rv_grid, ccf, peak_index):
    """Estimate the CCF peak RV between samples using a three-point parabola.

    Fit the selected local maximum and its two neighbours on an equally spaced
    RV grid. Return the parabola's vertex in the same units as rv_grid. This
    approximates the peak position.
    Return the selected sample's RV if a neighbour is missing or a is zero.
    """
    if peak_index <= 0 or peak_index >= ccf.size - 1:
        return rv_grid[peak_index]
    x0, x1, x2 = rv_grid[peak_index - 1], rv_grid[peak_index], rv_grid[peak_index + 1]
    y0, y1, y2 = ccf[peak_index - 1], ccf[peak_index], ccf[peak_index + 1]
    # Approximate the three CCF samples by y = a*u**2 + b*u + c,
    # where u = -1, 0, +1 at the left, central and right samples.
    a = 0.5 * (y0 - 2 * y1 + y2)
    b = 0.5 * (y2 - y0)
    if a == 0.0:
        return x1
    # The vertex is at u = -b/(2*a), relative to the central sample.
    peak_offset_bins = -b / (2 * a)
    # Keep the estimate within the two neighbouring samples.
    if peak_offset_bins < -1.0:
        peak_offset_bins = -1.0
    if peak_offset_bins > 1.0:
        peak_offset_bins = 1.0
    # Convert the offset to RV using the spacing of the supplied grid.
    rv_step = (x2 - x0) * 0.5
    return x1 + peak_offset_bins * rv_step


def find_common_wavelength_slice(wavelength, template_wavelength, rv_min, rv_max):
    """Select mock-spectrum wavelengths covered by the template at every trial RV.

    Doppler shifting changes the template's wavelength range. Trim the mock
    spectrum where needed so every trial from rv_min to rv_max can be evaluated
    without extrapolating the template flux. Use this same interval for all
    phases when measuring their relative RVs.

    Inputs are increasing wavelength arrays in Angstroms and RV limits in km/s.

    Returns:
        (start, stop): Indices selecting wavelength[start:stop]. The start sample is
        included and the stop sample is excluded. Apply the same slice to each
        phase spectrum.
    """
    c_kms = 299792.458
    # At the largest redshift the template's blue edge moves furthest right;
    # at the largest blueshift its red edge moves furthest left.
    covered_min = template_wavelength[0] * (1 + rv_max / c_kms)
    covered_max = template_wavelength[-1] * (1 + rv_min / c_kms)
    start = int(np.searchsorted(wavelength, covered_min, side="left"))
    stop = int(np.searchsorted(wavelength, covered_max, side="right"))
    if stop - start < 4:
        raise ValueError(
            "Template coverage leaves fewer than four usable wavelength points."
        )
    return start, stop


def coarse_to_fine_ccf_rv_search(
    wavelength,
    phase_spectra,
    template_wavelength,
    template_flux,
    *,
    half_range=40.0,
    coarse_step=1.0,
    fine_step=0.005,
    max_half_range=None,
    debug_callback=None,
):
    """Measure template-relative RVs using wide coarse and local fine searches.

    Doppler-shift the template and correlate it with every phase spectrum.
    First locate the most prominent peak on a coarse grid. Then search near
    that peak on a fine grid, select the local peak closest to the coarse peak,
    and refine its maximum with a three-point parabola.
    Phases sharing a coarse peak are batched so each shifted template is reused.

    Args:
        wavelength: Common spectrum wavelength grid in Angstroms.
        phase_spectra: Normalized flux array, shape (n_phases, n_wavelengths).
        template_wavelength, template_flux: Template wavelength and flux arrays.
        half_range: Symmetric RV search limit in km/s.
        coarse_step, fine_step: Trial RV spacing in km/s.
        max_half_range: Optional expansion limit for experiments. By default,
            the search range stays fixed; edge peaks are reported as failures.
        debug_callback: Optional consumer of the already calculated coarse/fine
            profiles, called once per phase after the selected search attempt.

    Returns:
        Raw RVs in km/s, in input phase order, and a diagnostic dictionary with
        wavelength slice bounds, search attempts and per-phase failures.
        Zero RV is valid; failed phases remain NaN. No phase mean is subtracted.
    """
    if max_half_range is None:
        max_half_range = half_range

    if not (0 < fine_step < coarse_step < half_range <= max_half_range):
        raise ValueError(
            "Require 0 < fine_step < coarse_step < half_range <= max_half_range."
        )
    wavelength = np.asarray(wavelength, dtype=np.float64)
    phase_spectra = np.asarray(phase_spectra, dtype=np.float64)
    template_wavelength = np.asarray(template_wavelength, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    if phase_spectra.ndim != 2 or phase_spectra.shape[1] != wavelength.size:
        raise ValueError("Spectra must have shape (n_phases, n_wavelengths).")
    if (
        not np.all(np.isfinite(phase_spectra))
        or not np.all(np.isfinite(template_flux))
        or not np.all(np.diff(wavelength) > 0)
        or not np.all(np.diff(template_wavelength) > 0)
    ):
        raise ValueError(
            "Search requires finite spectra and increasing wavelength grids."
        )
    n_phases = len(phase_spectra)
    if n_phases == 0:
        return np.empty(0), {"attempts": [], "failures": []}

    report = {"attempts": []}
    while True:
        start, stop = find_common_wavelength_slice(
            wavelength, template_wavelength, -half_range, half_range
        )
        selected_wavelength = wavelength[start:stop]
        selected_spectra = phase_spectra[:, start:stop]

        coarse_grid = np.arange(-half_range, half_range, coarse_step, dtype=np.float64)
        coarse_ccfs = calculate_ccf_batch(
            selected_wavelength,
            selected_spectra,
            template_wavelength,
            template_flux,
            -half_range,
            half_range,
            coarse_step,
        )
        if coarse_ccfs is None:
            raise ValueError(
                "Template coverage check failed after common-interval selection."
            )
        # Initialize every phase's peak index to -1 (no valid peak found yet).
        # Replace it with the coarse-grid index when a valid peak is found.
        coarse_peak_indices = np.full(n_phases, -1, dtype=int)
        # Start with no failure reason for each phase. Failed searches record
        # a reason below; successful searches leave the string empty.
        failure_reasons = [""] * n_phases

        for phase_index, curve in enumerate(coarse_ccfs):
            span = np.ptp(curve)
            if not np.isfinite(span) or span == 0:
                failure_reasons[phase_index] = "flat_ccf"
                continue
            # An edge maximum cannot be accepted as a measured RV.
            if np.argmax(curve) in (0, len(curve) - 1):
                failure_reasons[phase_index] = "coarse_edge"
                continue
            # Normalize only for prominence selection; refinement uses raw CCFs.
            candidates, properties = find_peaks(
                (curve - curve.min()) / span,
                prominence=0.1,
            )
            if len(candidates):
                coarse_peak_indices[phase_index] = candidates[
                    np.argmax(properties["prominences"])
                ]
            else:
                failure_reasons[phase_index] = "no_prominent_peak"
        report["attempts"].append(
            {
                "half_range_kms": half_range,
                "coarse_rv_points": len(coarse_grid),
                "failed_phases": int(np.sum(coarse_peak_indices < 0)),
            }
        )
        # A wider search may reveal an edge or missing peak, but cannot fix a
        # flat CCF. Retry all phases on the newly covered wavelength interval.
        if (
            any(
                reason in ("coarse_edge", "no_prominent_peak")
                for reason in failure_reasons
            )
            and half_range < max_half_range
        ):
            half_range = min(half_range * 2, max_half_range)
            continue
        break

    rv_curve = np.full(n_phases, np.nan)

    def export_profile(phase_index, fine_grid=None, fine_ccf=None):
        # Consume the current profiles without saving additional CCF arrays.
        coarse_index = coarse_peak_indices[phase_index]
        debug_callback(
            phase_index=int(phase_index),
            coarse_grid=coarse_grid,
            coarse_ccf=coarse_ccfs[phase_index],
            coarse_peak_rv=coarse_grid[coarse_index] if coarse_index >= 0 else None,
            fine_grid=fine_grid,
            fine_ccf=fine_ccf,
            raw_rv=rv_curve[phase_index],
            failure_reason=failure_reasons[phase_index],
        )

    if debug_callback is not None:
        for phase_index in np.flatnonzero(coarse_peak_indices < 0):
            export_profile(phase_index)
    fine_groups = []
    # Group by coarse peak location, not phase order. This is serial batching,
    # not process parallelism; each group shares the same shifted templates
    for coarse_index in np.unique(coarse_peak_indices[coarse_peak_indices >= 0]):
        phase_indices = np.flatnonzero(coarse_peak_indices == coarse_index)
        # Search between the neighbouring coarse samples. np.arange excludes
        # its stop, so add one fine step to include the right neighbour when
        # possible. Only interior local peaks can be refined below.
        rv_min = max(-half_range, coarse_grid[coarse_index] - coarse_step)
        rv_max = min(
            half_range,
            coarse_grid[coarse_index] + coarse_step + fine_step,
        )
        fine_grid = np.arange(rv_min, rv_max, fine_step, dtype=np.float64)
        fine_ccfs = calculate_ccf_batch(
            selected_wavelength,
            selected_spectra[phase_indices],
            template_wavelength,
            template_flux,
            rv_min,
            rv_max,
            fine_step,
        )
        if fine_ccfs is None:
            raise ValueError("Fine search exceeds the selected template coverage.")

        # Follow the peak selected by coarse prominence: choose the local peak
        # closest to its coarse RV, rather than a higher neighbouring peak.
        # Refine that peak using its two adjacent samples.
        for phase_index, curve in zip(phase_indices, fine_ccfs):
            candidates, _ = find_peaks(curve)
            if not len(candidates):
                failure_reasons[phase_index] = (
                    "fine_edge"
                    if np.argmax(curve) in (0, len(curve) - 1)
                    else "no_fine_peak"
                )
                if debug_callback is not None:
                    export_profile(phase_index, fine_grid, curve)
                continue
            # Measure each local peak's RV distance from the selected coarse peak.
            distances = np.abs(fine_grid[candidates] - coarse_grid[coarse_index])
            # Follow the nearest peak, even if another peak is higher.
            peak_index = candidates[np.argmin(distances)]
            # Fit this peak and its two neighbours to refine the RV between samples.
            rv_curve[phase_index] = _refine_ccf_peak(fine_grid, curve, peak_index)
            if debug_callback is not None:
                export_profile(phase_index, fine_grid, curve)
        fine_groups.append(
            {
                "center_kms": float(coarse_grid[coarse_index]),
                "phases": len(phase_indices),
                "fine_rv_points": len(fine_grid),
            }
        )
    report.update(
        wavelength_start=start,
        wavelength_stop=stop,
        usable_wavelength_points=stop - start,
        fine_groups=fine_groups,
        failures=failure_reasons,
        half_range_kms=half_range,
    )
    return rv_curve, report


@dataclass
class RVResult:
    """RV in km/s, search diagnostics, and optional final-attempt CCF profiles.

    rv is a float for a single input spectrum, otherwise a 1D array. profiles
    is None unless requested; otherwise it is a list in input spectrum order.
    Each profile has separate coarse/fine grids and curves. Failed spectra may
    have no fine profile. No additional full-range fine CCF is calculated.
    """

    rv: float | np.ndarray
    report: dict
    profiles: list[dict] | None = None


def _validate_inputs(wavelength, spectra, template_wavelength, template_flux):
    for name, grid in (("Target", wavelength), ("Template", template_wavelength)):
        if (
            grid.ndim != 1
            or grid.size < 4
            or not np.isfinite(grid).all()
            or np.any(grid <= 0)
            or not np.all(np.diff(grid) > 0)
        ):
            raise ValueError(
                f"{name} wavelength must have at least four finite, "
                "positive, strictly increasing samples."
            )
    if spectra.ndim != 2 or spectra.shape[1] != wavelength.size:
        raise ValueError(
            "Target flux must have shape (n_wavelengths,) or "
            "(n_spectra, n_wavelengths)."
        )
    if (
        template_flux.ndim != 1
        or template_flux.shape != template_wavelength.shape
        or not np.isfinite(template_flux).all()
        or not np.isfinite(spectra).all()
    ):
        raise ValueError(
            "Fluxes must be finite; template flux must match its wavelength."
        )


def measure_rv(
    wavelength,
    flux,
    template_wavelength,
    template_flux,
    *,
    ccf_rv_oversample_factor=50.0,
    rv_half_range=40.0,
    n_retries=1,
    return_ccf=False,
    spectrum_vel_step=None,
    debug_callback=None,
):
    """Measure template-relative RVs from continuum-flattened spectra.

    Args:
        wavelength: Target wavelengths in Angstroms, shape (n_pixels,).
            Use one continuous band/order with at least four finite, positive,
            strictly increasing samples; uniform spacing is not required.
        flux: Finite, flattened target flux, shape (n_pixels,) or
            (n_spectra, n_pixels) for spectra sharing the wavelength grid.
        template_wavelength: Template wavelengths in Angstroms, shape
            (n_template_pixels,), meeting the same wavelength requirements.
            Use the same air/vacuum convention and velocity frame as the target.
        template_flux: Finite, flattened template flux matching its wavelength
            array. Target and template may have different sampling.
        ccf_rv_oversample_factor: Fine-search oversampling factor, finite and
            at least 1 (default 50).
        rv_half_range: Initial search limit on either side of zero, in km/s
            (default 40); must exceed the 1 km/s coarse step.
        n_retries: Maximum extra attempts (default 1). Each retry doubles the
            range and reruns all spectra on a new common wavelength interval.
        return_ccf: Retain final-attempt coarse and local fine profiles
            (default False).
        spectrum_vel_step: Optional spectral sampling in km/s. If None,
            estimate as c * median(diff(log(wavelength))). Supplying a value
            skips estimation; EXSPLOSYN supplies its original sampling.
        debug_callback: Optional callable receiving attempt_index, phase_index
            (input row), coarse_grid/coarse_ccf, fine_grid/fine_ccf,
            coarse_peak_rv, raw_rv and failure_reason as keyword arguments.
            Both indices are zero based; fine profiles may be None.

    Returns:
        RVResult: rv in km/s (float for one spectrum, 1D array for a batch),
        report with search settings, wavelength bounds and failure reasons,
        and profiles in input order when requested (otherwise None).
        Positive RV shifts the template redward to match the target.

    Notes:
        Fine step = spectrum_vel_step / ccf_rv_oversample_factor; it must be
        positive and smaller than the fixed 1 km/s coarse step. Inferred
        sampling sets search spacing, not RV precision. No normalization,
        target resampling, frame correction or mean subtraction is applied.
        Only coarse-edge or missing-prominent-peak failures trigger retries.
        Failures without a retry return NaN; unresolved failures after retrying
        raise ValueError. Insufficient template coverage also raises ValueError.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    template_wavelength = np.asarray(template_wavelength, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    single = flux.ndim == 1
    # np.newaxis adds a new dimension to the flux array, making it 2D if it's a single spectrum.
    spectra = flux[np.newaxis, :] if single else flux

    _validate_inputs(wavelength, spectra, template_wavelength, template_flux)
    if (
        isinstance(n_retries, (bool, np.bool_))
        or not isinstance(n_retries, (int, np.integer))
        or n_retries < 0
    ):
        raise ValueError("n_retries must be a nonnegative integer.")
    if not np.isfinite(ccf_rv_oversample_factor) or ccf_rv_oversample_factor < 1:
        raise ValueError("ccf_rv_oversample_factor must be finite and at least 1.")

    inferred = spectrum_vel_step is None
    if inferred:
        spectrum_vel_step = 299792.458 * np.median(np.diff(np.log(wavelength)))
    if not np.isfinite(spectrum_vel_step) or spectrum_vel_step <= 0:
        raise ValueError("spectrum_vel_step must be finite and positive.")
    fine_step = float(spectrum_vel_step / ccf_rv_oversample_factor)
    if not (np.isfinite(rv_half_range) and 0 < fine_step < 1.0 < rv_half_range):
        raise ValueError("Require 0 < fine_step < 1 km/s < rv_half_range, all finite.")

    profiles = None
    attempts = []
    half_range = float(rv_half_range)
    for attempt in range(n_retries + 1):
        profiles = [None] * len(spectra) if return_ccf else None

        def consume_profile(*, phase_index, **profile):
            if profiles is not None:
                # Copies retain only explicitly requested final-attempt profiles.
                profiles[phase_index] = {
                    "phase_index": phase_index,
                    **{
                        key: value.copy() if isinstance(value, np.ndarray) else value
                        for key, value in profile.items()
                    },
                }
            if debug_callback is not None:
                debug_callback(
                    attempt_index=attempt, phase_index=phase_index, **profile
                )

        rv, report = coarse_to_fine_ccf_rv_search(
            wavelength,
            spectra,
            template_wavelength,
            template_flux,
            half_range=half_range,
            coarse_step=1.0,
            fine_step=fine_step,
            max_half_range=half_range,
            debug_callback=(
                consume_profile if return_ccf or debug_callback is not None else None
            ),
        )
        attempts.extend(report["attempts"])
        failures = report["failures"]
        # Match the existing outer EXSPLOSYN retry policy, including failures
        # discovered at the fine stage after a coarse-triggered retry.
        if attempt > 0 and any(failures):
            retryable = any(
                reason in ("coarse_edge", "no_prominent_peak") for reason in failures
            )
            if attempt == n_retries or not retryable:
                raise ValueError(
                    f"CCF peak search failed after retrying at +/-{half_range:g} km/s. "
                    f"Spectrum failure reasons: {failures}"
                )
        if not any(
            reason in ("coarse_edge", "no_prominent_peak") for reason in failures
        ):
            break
        if attempt == n_retries:
            break
        logger.warning(
            "CCF edge or missing peak within +/-%.1f km/s; "
            "retrying all spectra at +/-%.1f km/s.",
            half_range,
            half_range * 2,
        )
        half_range *= 2

    report.update(
        attempts=attempts,
        fine_step_kms=fine_step,
        coarse_step_kms=1.0,
        spectrum_vel_step_kms=float(spectrum_vel_step),
        sampling_inferred=inferred,
        retries_used=attempt,
        package_version=__version__,
    )
    return RVResult(float(rv[0]) if single else rv, report, profiles)
