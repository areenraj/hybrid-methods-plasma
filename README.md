## Overview

A 1D FVM solver using Landau-fluid closures. The codebase is split into three files:

- **`fv_core.py`** — pure numerics library (Grid, FluidSolver, time integrators).
- **`landau_damping.py`** — electron and ion acoustic Landau damping simulation.
- **`two_stream.py`** — two-stream instability simulation with two coupled electron beams.

## Landau damping (`landau_damping.py`)

Toggle between electron and ion acoustic damping with `ionDamping`.

**Electron damping** — use `k` around `0.3`–`0.5` and set the domain length accordingly (`L = 2pi/k`). At small `k` the electron Landau damping rate drops toward zero rapidly.

**Ion damping** — Much smaller `k` values are possible with the `k*lambda_D << 1` condition, here `tau` is the main factor on which the damping rate depends. In this case, I would recommend first setting the domain length `L` to a large value like 100, and then setting the `k = 2pi/L` value - basically the reverse of the electron damping case. Make sure to use a higher number of grid cells with larger domain. I would recommend using atleast `dx <= 0.5`.

Output saved to `results/efield_data_electron.npz` or `results/efield_data_ion.npz`.

## Two-stream instability (`two_stream.py`)

Two counter-propagating electron beams at ±`v_b` coupled through a shared electric field. Initial condition includes the beams with constant density beams with a velocity perturbations

**Plot modes** — set `plot_mode`:
- `"v1"` — velocity, density perturbation, electric field, total pressure, energy
- `"v2"` — per-beam density-velocity phase space plots, velocity difference, electric potential, energy

**Movie export** — set `save_movie = True`. Requires `ffmpeg`. Output saved to `results/two_stream.mp4`.

Output saved to `results/efield_data_twostream.npz`.

## Sweep scripts

**`tau_sweep/sweep_tau.sh`** — sweeps `tau` at fixed `k` for ion acoustic damping, extracts γ at each point, plots numerical vs analytical.

```bash
./tau_sweep/sweep_tau.sh   # edit K, TAU_MIN, TAU_MAX, N_POINTS inside first
```

**`electron_k_sweep/sweep_k.sh`** — sweeps `k` for electron Landau damping, extracts γ at each point, plots numerical vs `plasmadisp` analytical rate.

```bash
./electron_k_sweep/sweep_k.sh   # edit K_MIN, K_MAX, N_POINTS, BASE_NX inside first
```

Both scripts call `extract_gamma.py` to fit the exponential decay from the saved `.npz` file.

## Analysis

**`electron_damping_analysis.py`** — loads `results/efield_data_electron.npz`, extracts damping rate from the Fourier mode amplitude, plots decay and comparison against the `plasmadisp` Landau rate.

```bash
python3 electron_damping_analysis.py
```

To clean results: `rm results/*`

## 3–7 moment models vs NuFI (branch `testing`)

Comparison of the Landau-fluid models in `fv_core.py` (3HP, 4HP, 5-, 6-, 7-moment; see `docs/moment_closures.md`) against the kinetic GPU solver NuFI.

**Setup**
- Electron Landau damping: k = 0.23–0.65 (20 points). NuFI Nx=64, Nu=256, dt=0.1, T=60; fluid nx=256, T=60.
- Ion-acoustic damping: τ = 0.1–1.25 (16 points), k = 2π/100, Boltzmann electrons. NuFI Nx=32, Nu=128, dt=0.2, T=450; fluid nx=512, T=450.
- Two-stream: ±1 electron beams, n=0.5 and T=0.16 each, k=0.5. NuFI Nx=64, Nu=512, dt=0.1, T=40; fluid nx=128, T=40.
- Each run is compared with the exact kinetic root (plasma dispersion function).

**What was added (the fluid solver and drivers are unchanged)**
- `NumericalFlowIteration/validation/scripts/moments/`: `run_nufi.py` runs NuFI once per point and caches it; `run_fluid.py` + `fluid_worker.py` run all fluid models in parallel; `analyze.py` produces the fits, figures and tables. Usage is in its `README.md`.
- `validation/scripts/generate/gen_config_twostream.py`: generates NuFI's two-stream config (reproduces the earlier saved run).
- γ is now fitted from the envelope of the |E_k| peaks, with the same fitter for NuFI and fluid. The old fits regressed through the oscillation zeros and biased the saved NuFI rates (e.g. k=0.23: −0.0198 vs exact −0.0007).
- **Initial-condition matching (only inside `fluid_worker.py`, mode `matched`).** NuFI starts from (1+α cos kx)·Maxwellian, i.e. uniform T and p = T₀ρ, while `landau_damping.py` uses uniform p. For two-stream, NuFI seeds a density ripple (α=1e-3) while `two_stream.py` seeds a velocity ripple. Matching these removes the startup offset against NuFI. The sweeps use the drivers' own IC; the damping rate does not depend on it.

**Results** (`cachedResults/moments_vs_nufi_2026-11-19/`: figures, tables, raw NuFI `statistics.csv`, fluid `.npz` traces, logs)

| mean \|γ − γ_NuFI\|/\|γ_NuFI\| | 3HP | 4HP | 5M | 6M | 7M |
|---|---|---|---|---|---|
| electron, kλ_D ≥ 0.3 | 38.5% | 25.4% | 10.3% | 4.0% | 2.0% |
| ion, all τ | 56.5% | 37.3% | 8.5% | 2.4% | 1.9% |

- NuFI agrees with the exact roots to about 2%.
- Two-stream with the matched IC: all models follow NuFI through the linear phase (γ = 0.231–0.239 vs NuFI 0.231, exact 0.2385).
- The 6- and 7-moment two-stream runs go to NaN near saturation: 6M at t≈20.0 (driver IC) / 31.7 (matched); 7M at t≈21.4 / 33.5. The 6M NaN contradicts `docs/moment_closures.md` §8.3.

## Dependencies

- [`plasmadisp`](https://github.com/ergodicio/plasmadisp.git) — analytical electron Landau damping rates
- `ffmpeg` — movie export from `two_stream.py` (`brew install ffmpeg`)
