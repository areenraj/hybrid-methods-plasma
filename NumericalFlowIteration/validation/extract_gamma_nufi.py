#!/usr/bin/env python3
"""
Extract the damping/growth rate gamma from a NuFI GPU run's statistics.csv.

statistics.csv columns (semicolon-separated):
    Time; L1-Norm; L2-Norm; Electric Energy; Kinetic Energy; Total Energy; Entropy

gamma is extracted from the electric energy E_el = (1/2) * |E|^2:
    |E|^2 ~ exp(2*gamma*t)  =>  E_el ~ exp(2*gamma*t)
    => gamma = 0.5 * d(ln E_el)/dt

We fit ln(E_el) vs t over the linear (exponential) phase, identified as the
segment after the peak of E_el (damping) or after the initial transient
(growth), using a robust window selection.
"""

import sys
import os
import numpy as np


def load_statistics(path="statistics.csv"):
    data = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            parts = line.split(";")
            if len(parts) < 4:
                continue
            try:
                data.append([float(p.strip()) for p in parts])
            except ValueError:
                continue
    return np.array(data)


def extract_gamma(data):
    t    = data[:, 0]
    E_el = data[:, 3]   # Electric Energy = (1/2)|E|^2

    if len(t) < 10:
        return float("nan")

    # log electric energy: E_el ~ A * exp(2*gamma*t) => ln(E_el) linear in t
    mask = E_el > 0
    if mask.sum() < 10:
        return float("nan")

    t_m = t[mask]
    lnE = np.log(E_el[mask])

    # Find peak of electric energy
    i_peak = np.argmax(E_el[mask])

    # For damping (peak not at end): fit from peak onward
    # For growth   (peak at end):    fit from ~10% of run onward to avoid transient
    n = len(t_m)
    if i_peak > n * 0.7:
        # Looks like growth — fit the exponential rise after the initial transient
        i_start = max(1, n // 10)
        i_end   = i_peak + 1
    else:
        # Damping — fit the decay tail
        i_start = i_peak
        i_end   = n

    t_fit  = t_m[i_start:i_end]
    lE_fit = lnE[i_start:i_end]

    if len(t_fit) < 3:
        # Fallback: fit everything
        t_fit, lE_fit = t_m, lnE

    coeffs = np.polyfit(t_fit, lE_fit, 1)
    # slope of ln(E_el) = 2*gamma  =>  gamma = slope / 2
    gamma = coeffs[0] / 2.0
    return gamma


def main():
    # Allow an optional path argument; default to cwd/statistics.csv
    path = sys.argv[1] if len(sys.argv) > 1 else "statistics.csv"
    if not os.path.isfile(path):
        print(f"0.00000000", flush=True)
        sys.exit(0)

    data = load_statistics(path)
    if data.size == 0:
        print(f"0.00000000", flush=True)
        sys.exit(0)

    gamma = extract_gamma(data)
    print(f"{gamma:.8f}")


if __name__ == "__main__":
    main()
