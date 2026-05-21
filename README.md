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

## Dependencies

- [`plasmadisp`](https://github.com/ergodicio/plasmadisp.git) — analytical electron Landau damping rates
- `ffmpeg` — movie export from `two_stream.py` (`brew install ffmpeg`)
