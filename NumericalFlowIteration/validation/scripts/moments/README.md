# 3–7 moment models vs NuFI

Helper scripts that compare the Landau-fluid models on `dev`
(`hammett_perkins`, `4moment_…`, `5moment_…`, `6moment_…`, `7moment_…`) against NuFI.
They only *read* the drivers (`landau_damping.py`, `two_stream.py`, `fv_core.py`); nothing in them is modified.

```bash
cd NumericalFlowIteration/validation/scripts/moments
python3 run_nufi.py            # GPU: electron k sweep, ion tau sweep, two-stream (cached, ~40 min)
python3 run_fluid.py --jobs 12 # CPU: all fluid runs in parallel (cached)
python3 analyze.py             # fits + figures + tables -> validation/results/moments/
```

| Script | Role |
|---|---|
| `common.py` | sweep ranges, NuFI/fluid resolutions, colours, exact kinetic roots, fitters |
| `run_nufi.py` | patches `nufi/config.hpp` via `generate/gen_config_*.py`, rebuilds, runs `test_nufi_gpu_1d`, caches `statistics.csv`; restores `config.hpp` afterwards. NuFI runs once per point (it does not depend on the fluid model) |
| `fluid_worker.py` | one fluid run in a private temp dir; `asis` = driver IC, `matched` = NuFI's IC (monkeypatched in-process) |
| `run_fluid.py` | runs all fluid jobs in parallel |
| `analyze.py` | γ tables, figures, `summary.md` |

Raw traces are cached in `NumericalFlowIteration/bin/moments_cache/` (git-ignored; override with `MOMENTS_CACHE`).

**Fitting.** Both NuFI and fluid γ come from a linear fit of ln(peak |E_k|) (parabolic peak refinement),
starting at t = 5 (electron) / 40 (ion), stopping at the noise floor or the first rising peak.
The original fits (`extract_gamma_nufi.py`, `run_hp_*.py`) regress through the oscillation zeros and are
biased, most visibly for NuFI at small k with `T_END = 16`; they are reported as `legacy` for reference.

**Initial conditions.** NuFI starts from `(1 + α cos kx) · Maxwellian`, i.e. uniform temperature and
`p = T0 ρ`. `landau_damping.py` uses uniform `p = T0` (temperature perturbed by `−α cos kx`).
`two_stream.py` seeds a velocity perturbation, while NuFI seeds a density perturbation `α = 1e-3`.
The `matched` runs remove these differences without changing the drivers.
