# rv-ccf

A small CCF radial-velocity function for the authors' personal use and
collaborations. This code also serving as an RV submodule for the [EXSPLOSYN project](https://github.com/shihyuntang/EXSPLOSYN). 

The `rv-ccf` module is intended for simple use cases: 
applying the same CCF RV pipeline to simulated and observed
spectra for reducing measurement differences caused by using different pipelines.

--- 

The basic function is like:
```python
from rv_ccf import measure_rv

result = measure_rv(
    wavelength, 
    flux, 
    template_wavelength, 
    template_flux,
    ccf_rv_oversample_factor=50,
    rv_half_range=40,  # km/s on either side of zero
    n_retries=1,
    return_ccf=True,
)

rv = result.rv  # km/s, relative to the template
coarse_velocity = result.profiles[0]["coarse_grid"]
coarse_ccf = result.profiles[0]["coarse_ccf"]
```

- **Inputs:** continuum-flattened target and template fluxes with their wavelength arrays in Angstroms. 
  Wavelengths must be positive and strictly increasing; all inputs must be finite.
- **Shape:** one target spectrum or a batch sharing one wavelength grid; one 1D template. Pass one continuous band/order per call. Target and template may have different, nonuniform sampling.
- **Preparation:** the user handles file loading, flattening, template
  broadening, masking and frame corrections.
- **Sampling:** the coarse search step is 1 km/s. The fine step is the estimated
  spectral velocity spacing divided by `ccf_rv_oversample_factor` (default 50).
  Set `spectrum_vel_step` explicitly to skip estimation; EXSPLOSYN does this
  to preserve its existing search grids.
- **Retries:** `n_retries` counts extra attempts. A coarse-edge or missing
  prominent peak triggers range doubling and a new common wavelength interval
  for every spectrum. Failed measurements without a retry return NaN;
  unresolved failures after retrying raise `ValueError`.
- **Results:** `result.rv` is a float for one target or an array for a batch.
  Positive RV shifts the template redward to match the target. No mean RV is
  subtracted. `result.report` records search settings and failure reasons.
- **CCF output:** with `return_ccf=True`, `result.profiles` contains the final
  attempt's coarse and local fine grids and curves in input order. Fine profiles
  may be absent for failed measurements. No RV uncertainty is estimated.

## Installation

EXSPLOSYN installs `rv-ccf` automatically when creating its Conda environment;
no separate clone or installation is needed.

For use outside EXSPLOSYN, activate your Python environment and run:

```sh
python -m pip install "rv-ccf @ git+https://github.com/shihyuntang/rv-ccf.git@v0.1.0"
```

- Requires Python 3.11–3.13 and Git. Python 3.11 is the tested setup.
- Pip installs NumPy, SciPy and Numba if compatible versions are not present.

For local development or to run the tests:

```sh
git clone https://github.com/shihyuntang/rv-ccf.git
cd rv-ccf
git checkout v0.1.0
python -m pip install -e .
python -m unittest discover -s tests -v
```

- Omit `git checkout v0.1.0` to work on the default development branch.
- In an EXSPLOSYN environment with dependencies already installed, use
  `python -m pip install -e . --no-deps --no-build-isolation` for local edits.
- The tests compare fixed reference results and exercise sampling, retries and
  failure handling. Retry warnings are expected in several tests.
