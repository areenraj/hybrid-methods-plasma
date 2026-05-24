#!/usr/bin/env python3
"""
Run the Hammett-Perkins fluid model for electron Landau damping at a given k
and print the damping rate gamma.

Usage:
    python3 run_hp_electron.py <k> <nx> <t_end> [model]

model: "hammett_perkins" (3HP, default) or "4moment_hammett_perkins" (4HP)

Must be run from the NumericalFlowIteration directory; the parent directory
(which contains landau_damping.py) is added to sys.path automatically.
"""

import sys
import os
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: run_hp_electron.py <k> <nx> <t_end> [model]")
        sys.exit(1)

    k_val  = float(sys.argv[1])
    nx     = int(sys.argv[2])
    t_end  = float(sys.argv[3])
    model  = sys.argv[4] if len(sys.argv) > 4 else "hammett_perkins"

    # Add parent dir so we can import landau_damping
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))
    sys.path.insert(0, parent_dir)

    # Results must land in parent_dir/results/
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Change to parent dir so relative paths in landau_damping work
    os.chdir(parent_dir)

    import matplotlib
    matplotlib.use("Agg")
    import landau_damping as sim

    sim.k             = k_val
    sim.L             = 2.0 * np.pi / k_val
    sim.nx            = nx
    sim.ionDamping    = False
    sim.t_end         = t_end
    sim.MODEL         = model
    sim.plot_interval = int(1e9)   # disable plotting

    sim.main()

    # Extract gamma from saved npz
    data    = np.load("results/efield_data_electron.npz", allow_pickle=True)
    t       = data["t"]
    nx_data = int(data["nx"].item())
    E_kmode = data["E_kmode_real"] + 1j * data["E_kmode_imag"]
    amp     = np.abs(E_kmode) * 2.0 / nx_data

    i_peak        = np.argmax(amp)
    t_f, a_f      = t[i_peak:], amp[i_peak:]
    mask          = a_f > 0
    t_f, a_f      = t_f[mask], a_f[mask]

    if len(t_f) < 3:
        print("0.00000000")
        return

    coeffs    = np.polyfit(t_f, np.log(a_f), 1)
    gamma_fit = coeffs[0]
    print(f"{gamma_fit:.8f}")

if __name__ == "__main__":
    main()
