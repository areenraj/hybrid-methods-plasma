#!/usr/bin/env python3
"""
Side-by-side panel plot of |E_k(t)| for HP fluid and NuFI kinetic
for the two-stream instability.

Usage:
    python3 compare_nufi.py [--hp-file PATH] [--nufi-file PATH]

Defaults read from (and output saved to):
    validation/results/4moment_hammett_perkins/two_stream/
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.normpath(os.path.join(
    _SCRIPT_DIR, "..", "..", "results", "4moment_hammett_perkins", "two_stream"
))

GREEN  = "#6aA84F"
PURPLE = "#9B4E8F"
K      = 0.5
V_B    = 1.0


def load_hp(path):
    d   = np.load(path)
    t   = d["t"]
    nx  = int(d["nx"][0])
    E_k = np.abs(d["E_kmode_real"] + 1j * d["E_kmode_imag"]) * 2.0 / nx
    return t, E_k


def load_nufi(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            parts = line.split(";")
            if len(parts) < 8:
                continue
            try:
                rows.append([float(p.strip()) for p in parts])
            except ValueError:
                continue
    data = np.array(rows)
    return data[:, 0], data[:, 7]


def analytical_growth_rate(k=K, v_b=V_B):
    kv    = k * v_b
    kv2   = kv * kv
    roots = np.roots([1.0, 0.0, -(2*kv2 + 1), 0.0, kv2*kv2 - kv2])
    return float(np.max(roots.imag))


def _theory_line(t, amp, gamma):
    log_mid = np.mean(np.log(np.maximum(amp, 1e-300)))
    t_mid   = 0.5 * (t[0] + t[-1])
    A       = np.exp(log_mid - gamma * t_mid)
    return A * np.exp(gamma * t)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hp-file",
                        default=os.path.join(_DATA_DIR, "efield_data_twostream.npz"),
                        help="Path to HP .npz file")
    parser.add_argument("--nufi-file",
                        default=os.path.join(_DATA_DIR, "statistics.csv"),
                        help="Path to NuFI statistics.csv")
    args = parser.parse_args()

    t_hp,   E_hp   = load_hp(args.hp_file)
    t_nufi, E_nufi = load_nufi(args.nufi_file)
    gamma_c = analytical_growth_rate()
    print(f"Analytical cold-fluid growth rate: γ = {gamma_c:.4f}")

    fig, (ax_hp, ax_nufi) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r"Two-stream instability: $|E_k(t)|$   ($k=0.5$,  $v_b=1.0$,  $v_0/v_\mathrm{th}=2.5$)",
        fontsize=13,
    )

    ax_hp.semilogy(t_hp, E_hp, color=GREEN, lw=1.5)
    ax_hp.semilogy(t_hp, _theory_line(t_hp, E_hp, gamma_c),
                   ls=":", color="black", lw=1.5,
                   label=fr"cold-fluid $\gamma = {gamma_c:.3f}$")
    ax_hp.set_xlabel(r"$t$", fontsize=13)
    ax_hp.set_ylabel(r"$|E_k|$", fontsize=13)
    ax_hp.set_title("HP fluid (4-moment)", fontsize=13)
    ax_hp.tick_params(axis="both", labelsize=12)
    ax_hp.legend(fontsize=11)
    ax_hp.grid(which="both", ls="--", lw=0.4)

    ax_nufi.semilogy(t_nufi, E_nufi, color=PURPLE, lw=1.5)
    ax_nufi.semilogy(t_nufi, _theory_line(t_nufi, E_nufi, gamma_c),
                     ls=":", color="black", lw=1.5,
                     label=fr"cold-fluid $\gamma = {gamma_c:.3f}$")
    ax_nufi.set_xlabel(r"$t$", fontsize=13)
    ax_nufi.set_ylabel(r"$|E_k|$", fontsize=13)
    ax_nufi.set_title(r"NuFI kinetic (Vlasov–Poisson)", fontsize=13)
    ax_nufi.tick_params(axis="both", labelsize=12)
    ax_nufi.legend(fontsize=11)
    ax_nufi.grid(which="both", ls="--", lw=0.4)

    plt.tight_layout()
    os.makedirs(_DATA_DIR, exist_ok=True)
    outfile = os.path.join(_DATA_DIR, "compare_nufi.png")
    plt.savefig(outfile, dpi=180)
    print(f"Saved {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
