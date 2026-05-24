#!/usr/bin/env python3
"""
Compare NuFI GPU ion oscillation tau sweep against the analytical dispersion.

NOTE: NuFI uses the full Gauss-law Poisson (d²phi/dx² = delta_n_i), which
models ion plasma oscillations with a fixed electron background — NOT ion
acoustic waves. The analytical curve here uses the fluid ion-acoustic
dispersion (Chapurin Eq. 2.49) for reference, but the two physical models
differ; a perfect match is not expected.

Reads:  validation/nufi_ion_tau_results.txt   (tau  gamma)
Writes: validation/nufi_ion_tau_sweep.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def find_physical_root(tau, k, cs=1.0):
    """Fluid ion-acoustic dispersion (Chapurin 2.49)."""
    alpha = np.sqrt(8.0 / np.pi)
    a2k2  = alpha**2 * k**2 * cs**2
    k2cs2 = k**2 * cs**2

    c4 =  1.0
    c3 =  0.0
    c2 =  tau * a2k2 - k2cs2 * (1.0 + tau) - 2.0 * tau * k2cs2
    c1 =  1j * tau**1.5 * alpha**3 * k**3 * cs**3
    c0 = -k2cs2 * (1.0 + tau) * tau * a2k2

    roots  = np.roots([c4, c3, c2, c1, c0])
    damped = roots[roots.imag < 0]
    if len(damped) == 0:
        return None
    return damped[np.argmin(np.abs(damped.imag / (np.abs(damped.real) + 1e-30)))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=float, required=True,
                        help="Wavenumber used in the sweep")
    args = parser.parse_args()
    k = args.k

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, "nufi_ion_tau_results.txt")

    if not os.path.isfile(results_file):
        print(f"Results file not found: {results_file}")
        sys.exit(1)

    data = np.loadtxt(results_file)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    tau_num   = data[:, 0]
    gamma_num = data[:, 1]

    tau_analytic   = np.linspace(tau_num.min(), tau_num.max(), 300)
    gamma_analytic = np.zeros(300)
    for i, tau in enumerate(tau_analytic):
        root = find_physical_root(tau, k)
        if root is not None:
            gamma_analytic[i] = -root.imag

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tau_analytic, gamma_analytic, ls="-",  lw=2,   color="blue",
            label="Fluid analytical (Chapurin Eq. 2.49)")
    ax.plot(tau_num,     -gamma_num,      ls="--", lw=1.5, color="red")
    ax.scatter(tau_num,  -gamma_num,      s=60,            color="red",
               label="NuFI GPU (Gauss Poisson)", zorder=5)

    ax.set_xlabel(r"$\tau = T_{0i}/T_{0e}$")
    ax.set_ylabel(r"$-\gamma$ (damping rate)")
    ax.set_title(f"Ion oscillation damping rate  $k = {k}$  —  NuFI GPU\n"
                 r"(NuFI: Gauss Poisson; analytical: fluid ion-acoustic)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(which="both", ls="--", lw=0.5)

    plt.tight_layout()
    outfile = os.path.join(script_dir, f"nufi_ion_tau_k{k:.4f}.png")
    plt.savefig(outfile, dpi=200)
    print(f"Saved {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
