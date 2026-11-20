# 3–7 moment Landau-fluid models vs NuFI — cached results (2026-11-19)

Fluid code: branch `dev` @ 3ed892e (unmodified `fv_core.py`, `landau_damping.py`, `two_stream.py`).
NuFI: GPU build of the modified `test_nufi_gpu_1d` (8-column `statistics.csv`, `ion_acoustic` support).

## Layout

| Path | Contents |
|---|---|
| `figures_and_tables/` | all figures, `summary.md`, `gamma_electron.csv`, `gamma_ion.csv`, `twostream.csv`, `startup.csv` |
| `nufi/electron/statistics__<k>.csv` | NuFI electron Landau runs (k = 0.23…0.65 + 0.5). Nx=64, Nu=256, dt=0.1, T=60, v∈[−8,8], α=0.01 |
| `nufi/ion/statistics__<tau>.csv` | NuFI ion-acoustic runs (τ = 0.1…1.25 + 0.5), k=2π/100. Nx=32, Nu=128, dt=0.2, T=450, v∈±6√τ, α=0.01 |
| `nufi/twostream/statistics__0.000000.csv` | NuFI two-stream: ±1 beams, T=0.16, density seed α=1e-3, k=0.5. Nx=64, Nu=512, dt=0.1, T=40 |
| `fluid/<case>/<model>__<param>__<ic>.npz` | fluid |E_k(t)| traces. case ∈ electron (param=k), ion (param=τ), twostream (param=0) |
| scripts | not duplicated here: they live in `NumericalFlowIteration/validation/scripts/moments/` (+ `generate/gen_config_twostream.py`) |
| `nufi.log`, `fluid.log` | run logs (the KeyboardInterrupt/make traceback in `nufi.log` is a deliberate restart; all points completed) |

`ic` = `asis` (driver's own initial condition) or `matched` (NuFI's initial condition: p = T0·ρ for Landau,
density seed α=1e-3 with T=0.16 for two-stream). `matched` exists for the startup checks (electron k=0.5, ion τ=0.5, all models)
and for two-stream; the k/τ sweeps are `asis`.

**NuFI `statistics.csv`** (`;`-separated, 1 header line): Time; L1; L2; Electric Energy; Kinetic Energy; Total Energy; Entropy; E_kmode_amp.
Column 8 is the physical |E_k| of the fundamental mode, directly comparable to the fluid `amp`.

**Fluid `.npz`** keys: `t`, `amp` (=|E_k|, same normalisation as NuFI col 8; t=0 included), `nan_time` (last finite t, NaN if
the run completed), `wall` [s], `n_steps`, `model`, `case`, `param`, `ic`, `nx`. Traces with >40000 steps are decimated.
Fluid resolution: electron nx=256 T=60, ion nx=512 T=450, two-stream nx=128 T=40; RK4, MUSCL/Rusanov, driver CFL.

## Loading

```python
import numpy as np
d = np.load("fluid/electron/6moment_hammett_perkins__0.500000__asis.npz"); t, amp = d["t"], d["amp"]
n = np.loadtxt("nufi/electron/statistics__0.500000.csv", delimiter=";", skiprows=1); tn, ampn = n[:, 0], n[:, 7]
```

## Regenerating the figures (no simulation re-run)

From the repo root:

```bash
D=$PWD/cachedResults/moments_vs_nufi_2026-11-19
MOMENTS_CACHE=$D MOMENTS_RESULTS=$D/figures_regen python3 NumericalFlowIteration/validation/scripts/moments/analyze.py
```

## Caveats

- γ fit = linear fit of ln(peak |E_k|), from t=5 (electron) / t=40 (ion), stopping at a noise floor of 1e-8·max or at the first
  rising peak. NuFI's own noise floor (~1e-10) is above that, so for strongly damped ion cases (τ ≳ 0.9) a few NuFI peaks in
  the noise enter the fit (e.g. τ=1.25: NuFI −0.0708 vs exact −0.0663). See `fit_check_ion.png`.
- 6- and 7-moment two-stream runs go to NaN (6M: t≈20.0 as-is / 31.7 matched; 7M: 21.4 / 33.5). The 6M NaN was confirmed with
  the unmodified `two_stream.py`; this contradicts the 6M "no NaN" entry in `docs/moment_closures.md` §8.3.
