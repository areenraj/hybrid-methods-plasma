#!/usr/bin/env python3
"""
Run ONE fluid case and save a compact |E_k(t)| trace.

    python3 fluid_worker.py <case> <model> <param> <ic> <out.npz>

case  : electron (param = k) | ion (param = tau) | twostream (param ignored, use 0)
ic    : asis    -> the driver's own initial condition, unchanged
        matched -> the initial condition NuFI actually uses (see below); applied by
                   monkeypatching the driver *in this process only*, the driver file is untouched

What "matched" changes
  electron / ion : NuFI starts from (1 + a cos kx) * Maxwellian(T0), i.e. uniform temperature and
                   p = T0 * rho.  The driver uses uniform p = T0 (so T is perturbed by -a cos kx).
                   matched sets p = T0 * rho and the consistent r = 3p^2/rho, X6 = 15p^3/rho^2.
  twostream      : NuFI seeds a density perturbation alpha = 0.001 on both beams (u = +-1,
                   T = 0.16 uniform).  The driver seeds a velocity perturbation u = +-(1 + 0.01 cos kx).

Each run executes inside a private temp directory, so runs can go in parallel without clobbering
the driver's results/*.npz output.
"""

import os
import sys
import shutil
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def _maxwellian_rows(U, rho, p):
    """Overwrite pressure and the Maxwellian higher moments of a state array in place."""
    U[2] = p
    n = U.shape[0]
    if n >= 4:
        U[3] = 0.0
    if n >= 5:
        U[4] = 3.0 * p**2 / rho
    if n >= 6:
        U[5] = 0.0
    if n >= 7:
        U[6] = 15.0 * p**3 / rho**2
    return U


def main():
    case, model, param, ic, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, C.REPO)
    from fv_core import Grid

    work = tempfile.mkdtemp(prefix=f"moments_{case}_")
    os.makedirs(os.path.join(work, "results"))
    os.chdir(work)

    try:
        if case in ("electron", "ion"):
            import landau_damping as sim
            sim.MODEL = model
            sim.plot_interval = int(1e9)
            if case == "electron":
                sim.ionDamping = False
                sim.k = param
                sim.L = 2.0 * np.pi / param
                sim.nx = C.FLUID_ELECTRON["nx"]
                sim.t_end = C.FLUID_ELECTRON["t_end"]
                T0 = 1.0
            else:
                sim.ionDamping = True
                sim.tau = param
                sim.k = C.ION_K
                sim.L = 2.0 * np.pi / C.ION_K
                sim.nx = C.FLUID_ION["nx"]
                sim.t_end = C.FLUID_ION["t_end"]
                T0 = param

            orig_ic = sim.initial_condition
            if ic == "matched":
                def patched(x_arr, mdl, k_val, tau_val, ion_damping, gamma_val):
                    U = orig_ic(x_arr, mdl, k_val, tau_val, ion_damping, gamma_val)
                    return _maxwellian_rows(U, U[0], T0 * U[0])
                sim.initial_condition = patched

            grid = Grid(sim.nx, sim.L)
            U0 = sim.initial_condition(grid.x, model, sim.k, sim.tau, sim.ionDamping, sim.gamma)
            E0 = sim._compute_E(grid, U0[0], sim.background_charge)
            fname = "results/efield_data_ion.npz" if case == "ion" else "results/efield_data_electron.npz"
            run = sim.main
        elif case == "twostream":
            import two_stream as ts
            ts.MODEL = model
            ts.nx = C.FLUID_TWOSTREAM["nx"]
            ts.t_end = C.FLUID_TWOSTREAM["t_end"]
            ts.plot_interval = 10**9
            ts.save_movie = False
            cfg = C.NUFI_TWOSTREAM
            orig_ic = ts.initial_conditions
            if ic == "matched":
                def patched_ts(x_arr, mdl, k_val, v_b_val, eps=0.01):
                    U1, U2 = orig_ic(x_arr, mdl, k_val, v_b_val, eps)
                    rho = 0.5 * (1.0 + cfg["alpha"] * np.cos(k_val * x_arr))
                    for U, sgn in ((U1, +1.0), (U2, -1.0)):
                        U[0] = rho
                        U[1] = rho * sgn * v_b_val
                        _maxwellian_rows(U, rho, cfg["vth2"] * rho)
                    return U1, U2
                ts.initial_conditions = patched_ts
            grid = Grid(ts.nx, ts.L)
            U1, U2 = ts.initial_conditions(grid.x, model, ts.k, ts.v_b)
            E0 = grid.poisson_e(U1[0] + U2[0])
            fname = "results/efield_data_twostream.npz"
            run = ts.main
        else:
            raise ValueError(case)

        t_wall = time.time()
        with np.errstate(all="ignore"):
            run()
        t_wall = time.time() - t_wall

        d = np.load(fname, allow_pickle=True)
        nx = int(np.ravel(d["nx"])[0])
        t = np.concatenate([[0.0], d["t"]])
        amp = np.concatenate([[np.abs(np.fft.fft(E0)[1]) * 2.0 / nx],
                              np.abs(d["E_kmode_real"] + 1j * d["E_kmode_imag"]) * 2.0 / nx])
        bad = ~np.isfinite(amp) | ~np.isfinite(t)
        nan_time = float(t[np.argmax(bad)]) if bad.any() else np.nan
        if bad.any():   # last finite time is the meaningful one when t itself went NaN
            fin = np.where(~bad)[0]
            nan_time = float(t[fin[-1]]) if len(fin) else 0.0
        n_steps = len(d["t"])
        # decimate long traces (keep <= 40000 samples) -- peaks stay well resolved
        stride = max(1, len(t) // 40000)
        keep = np.unique(np.concatenate([np.arange(0, len(t), stride), [len(t) - 1]]))
        np.savez_compressed(out, t=t[keep], amp=amp[keep], nan_time=nan_time, wall=t_wall,
                            n_steps=n_steps, model=model, case=case, param=param, ic=ic, nx=nx)
        print(f"{case} {model} {param} {ic}: steps={n_steps} wall={t_wall:.1f}s nan_t={nan_time}")
    finally:
        os.chdir("/")
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
