`Finite_Volume.py` simulates both electron and ion acoustic Landau damping. Set `ionDamping = True` for ion acoustic mode; leave it `False` for electron Langmuir mode. Outputs are saved to `results/`. 

## Ion damping (`ionDamping = True`)

Switches from electron to ion Landau damping. Also set `tau` (ion-to-electron temperature ratio) to desired value before running. This mode only works with the 3HP and 4HP models selected.

## Usage notes

**Electron damping** — use `k` around `0.3`–`0.5` and set the domain length accordingly (`L = 2pi/k`). At small `k` the electron Landau damping rate drops toward zero rapidly.

**Ion damping** — Much smaller `k` values are possible with the `k*lambda_D << 1` condition, here `tau` is the main factor on which the damping rate depends. In this case, I would recommend first setting the domain length `L` to a large value like 100, and then setting the `k = 2pi/L` value - basically the reverse of the electron damping case. Make sure to use a higher number of grid cells with larger domain. I would recommend using atleast `dx <= 0.5`.

## Analysis

**`electron_damping_analysis.py`** — loads `results/efield_data_electron.npz`, extracts gamma from the Fourier mode amplitude, and plots the decay + comparison against the Landau rate from `plasmadisp`.

```bash
python3 electron_damping_analysis.py
```

**`tau_sweep/sweep_tau.sh`** — sweeps `tau` at fixed `k`, runs the sim at each point, extracts γ via `extract_gamma.py`, and plots numerical vs analytical damping rate with `ion_tau_plot.py`.

```bash
./tau_sweep/sweep_tau.sh   # edit K, TAU_MIN, TAU_MAX, N_POINTS inside first
```

**`electron_k_sweep/sweep_k.sh`** — sweeps `k` for electron damping, runs the sim at each point, extracts γ via `extract_gamma.py`, and plots numerical vs plasmadisp Landau rate with `electron_k_plot.py`.

```bash
./electron_k_sweep/sweep_k.sh   # edit K_MIN, K_MAX, N_POINTS, BASE_NX inside first
```
To clean directories just `rm results/*`

## Dependencies

- [`plasmadisp`](https://github.com/ergodicio/plasmadisp.git) — for calculating the analytical electron landau damping rate
