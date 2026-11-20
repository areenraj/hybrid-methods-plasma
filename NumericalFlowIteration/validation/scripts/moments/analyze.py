#!/usr/bin/env python3
"""
Fit damping/growth rates from the cached NuFI and fluid runs and make the comparison figures.

    python3 analyze.py

Writes to validation/results/moments/:
  landau_electron_gamma.png   electron Landau damping rate vs k: exact, NuFI, 3-7 moment models
  landau_ion_gamma.png        ion-acoustic damping rate vs tau:  exact, NuFI, 3-7 moment models
  twostream_nufi_vs_moments.png   |E_k(t)| NuFI vs 4-7 moment models, driver IC as-is | IC matched
  startup_electron.png / startup_ion.png   early |E_k(t)|, driver IC as-is vs matched to NuFI
  fit_check_electron.png / fit_check_ion.png   peaks used by the envelope fit
  gamma_electron.csv, gamma_ion.csv, twostream.csv, startup.csv, summary.md
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12, "legend.fontsize": 9.5,
    "axes.edgecolor": C.INK2, "axes.labelcolor": "#0b0b0b", "xtick.color": C.INK2,
    "ytick.color": C.INK2, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dcdbd5", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2.0,
})


def _have(path):
    return os.path.exists(path)


# ------------------------------------------------------------------
# Landau damping sweeps
# ------------------------------------------------------------------
def landau_table(case):
    params = C.ELECTRON_K if case == "electron" else C.ION_TAU
    t_min = C.FIT_TMIN_ELECTRON if case == "electron" else C.FIT_TMIN_ION
    exact = (lambda p: C.exact_electron(p).imag) if case == "electron" else (lambda p: C.exact_ion(p).imag)
    rows = []
    for p in params:
        row = {"param": p, "exact": exact(p), "nufi": np.nan, "nufi_legacy": np.nan, "nufi_npk": 0}
        sp = C.nufi_stats_path(case, p)
        if _have(sp):
            d = C.load_nufi(sp)
            g, tp, _ = C.envelope_gamma(d[:, 0], d[:, 7], t_min)
            row.update(nufi=g, nufi_npk=len(tp), nufi_legacy=C.legacy_gamma_nufi(d))
        for m in C.MODELS_LANDAU:
            fp = C.fluid_trace_path(case, m, p, "asis")
            row[m] = row[m + "_legacy"] = np.nan
            row[m + "_npk"] = 0
            if _have(fp):
                d = C.load_fluid(fp)
                g, tp, _ = C.envelope_gamma(d["t"], d["amp"], t_min)
                row[m], row[m + "_npk"] = g, len(tp)
                row[m + "_legacy"] = C.legacy_gamma_fluid(d["t"], d["amp"])
        rows.append(row)
    return rows


def write_csv(rows, path, keys):
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.8g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


def plot_landau(rows, case, outfile):
    x = np.array([r["param"] for r in rows])
    fig, (ax, ae) = plt.subplots(2, 1, figsize=(8.2, 7.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 2], hspace=0.08))
    xs = np.linspace(x.min(), x.max(), 300)
    if case == "electron":
        ex = np.array([-C.exact_electron(v).imag for v in xs])
        xlabel, title = r"$k\lambda_D$", r"Electron Landau damping: NuFI vs 3–7 moment Landau-fluid models"
        ylabel = r"damping rate $-\gamma/\omega_{pe}$"
    else:
        ex = np.array([-C.exact_ion(v).imag for v in xs])
        xlabel = r"$\tau = T_{0i}/T_{0e}$"
        title = r"Ion-acoustic Landau damping ($k=2\pi/100$): NuFI vs 3–7 moment models"
        ylabel = r"damping rate $-\gamma$"
    ax.plot(xs, ex, color=C.EXACT_COLOR, lw=2, label="exact kinetic root", zorder=1)
    g_n = np.array([r["nufi"] for r in rows])
    for m in C.MODELS_LANDAU:
        g = np.array([r[m] for r in rows])
        ax.plot(x, -g, color=C.COLOR[m], lw=1.2, alpha=0.9, zorder=2)
        ax.plot(x, -g, ls="none", marker=C.MARKER[m], ms=6.5, mfc=C.COLOR[m], mec="#fcfcfb", mew=1.0,
                label=C.LABEL[m], zorder=3)
        err = 100.0 * (np.abs(g) - np.abs(g_n)) / np.abs(g_n)
        ae.plot(x, err, color=C.COLOR[m], lw=1.2, marker=C.MARKER[m], ms=5.5, mec="#fcfcfb", mew=0.8)
    ax.plot(x, -g_n, ls="none", marker="o", ms=8.5, mfc="none", mec=C.NUFI_COLOR, mew=1.8,
            label="NuFI (kinetic, GPU)", zorder=4)
    ex_pts = np.array([r["exact"] for r in rows])
    ae.plot(x, 100.0 * (np.abs(ex_pts) - np.abs(g_n)) / np.abs(g_n), color=C.EXACT_COLOR, lw=2,
            label="exact root")
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend(loc="lower right", frameon=False)
    ae.axhline(0.0, color=C.NUFI_COLOR, lw=1.0)
    ae.set_ylabel(r"$(|\gamma| - |\gamma_{\rm NuFI}|)/|\gamma_{\rm NuFI}|$  [%]")
    ae.set_xlabel(xlabel)
    ae.grid(True)
    lim = 60 if case == "electron" else 50
    ae.set_ylim(-lim, lim)
    ae.text(0.01, 0.04, "below 0: model under-damps", transform=ae.transAxes, color=C.INK2, fontsize=9)
    if case == "electron":
        for a in (ax, ae):
            a.axvspan(x.min(), 0.3, color="#efeee9", zorder=0, lw=0)
        ax.text(0.235, ax.get_ylim()[0] * 1.6, r"$k\lambda_D<0.3$: weakly damped,"
                "\nno Padé closure resolves it", fontsize=8.5, color=C.INK2, va="bottom")
    fig.savefig(outfile, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fit_check(case, outfile):
    params = [C.ELECTRON_K[0], C.ELECTRON_K[9], C.ELECTRON_K[-1]] if case == "electron" else \
             [C.ION_TAU[0], C.ION_TAU[7], C.ION_TAU[-1]]
    t_min = C.FIT_TMIN_ELECTRON if case == "electron" else C.FIT_TMIN_ION
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for ax, p in zip(axs, params):
        series = []
        sp = C.nufi_stats_path(case, p)
        if _have(sp):
            d = C.load_nufi(sp)
            series.append(("NuFI", C.NUFI_COLOR, d[:, 0], d[:, 7]))
        for m in C.MODELS_LANDAU:
            fp = C.fluid_trace_path(case, m, p, "asis")
            if _have(fp):
                d = C.load_fluid(fp)
                series.append((C.SHORT[m], C.COLOR[m], d["t"], d["amp"]))
        for name, col, t, a in series:
            g, tp, lp = C.envelope_gamma(t, a, t_min)
            ax.semilogy(t, a, color=col, lw=0.8, alpha=0.7)
            ax.semilogy(tp, np.exp(lp), ls="none", marker="o", ms=3.5, color=col,
                        label=f"{name}  γ={g:.4f}")
        ax.set_title(("k" if case == "electron" else "τ") + f" = {p:g}")
        ax.set_xlabel("t")
        ax.grid(True, which="major")
        ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel(r"$|E_k|$  (dots: peaks used in fit)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Startup diagnostic (driver IC vs NuFI-matched IC)
# ------------------------------------------------------------------
def startup(case, outfile):
    p = C.DIAG_K if case == "electron" else C.DIAG_TAU
    t_show = 30.0 if case == "electron" else 300.0
    sp = C.nufi_stats_path(case, p)
    if not _have(sp):
        return []
    nd = C.load_nufi(sp)
    tn, an = nd[:, 0], nd[:, 7]
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.9), sharey=True)
    rows = []
    for ax, m in zip(axs, C.MODELS_LANDAU):
        ax.semilogy(tn, an, color=C.NUFI_COLOR, lw=2.2, label="NuFI")
        for ic, ls, lab in (("asis", (0, (4, 2)), "driver IC (p uniform)"),
                            ("matched", "-", r"matched IC ($p=T_0\rho$)")):
            fp = C.fluid_trace_path(case, m, p, ic)
            if not _have(fp):
                continue
            d = C.load_fluid(fp)
            ax.semilogy(d["t"], d["amp"], color=C.COLOR[m], ls=ls, lw=1.6, label=lab)
            # deviation from NuFI, in units of the initial amplitude, over the first part of the run
            w = tn <= t_show
            af = np.interp(tn[w], d["t"], d["amp"])
            dev = np.max(np.abs(af - an[w])) / an[0]
            # envelope: first 3 peaks after t=0
            tpf, lpf = C.refined_peaks(d["t"], d["amp"])
            tpn, lpn = C.refined_peaks(tn, an)
            n3 = min(3, len(lpf), len(lpn))
            pk = 100.0 * np.max(np.abs(np.exp(lpf[:n3] - lpn[:n3]) - 1.0)) if n3 else np.nan
            rows.append(dict(case=case, param=p, model=m, ic=ic, max_dev_over_E0=dev,
                             first3_peak_err_pct=pk))
        ax.set_xlim(0, t_show)
        ax.set_title(C.LABEL[m], fontsize=10.5)
        ax.set_xlabel("t")
        ax.grid(True, which="major")
    axs[0].set_ylabel(r"$|E_k(t)|$")
    axs[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    lab = f"electron Landau, k = {p:g}" if case == "electron" else f"ion acoustic, τ = {p:g}, k = 2π/100"
    fig.suptitle(f"Startup check ({lab}): driver initial condition vs initial condition matched to NuFI",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return rows


# ------------------------------------------------------------------
# Two-stream
# ------------------------------------------------------------------
def twostream(outfile):
    sp = C.nufi_stats_path("twostream", 0.0)
    if not _have(sp):
        return []
    nd = C.load_nufi(sp)
    tn, an = nd[:, 0], nd[:, 7]
    gk = C.exact_twostream().imag
    fn = C.growth_fit(tn, an)
    rows = [dict(model="NuFI", ic="-", **fn, nan_time=np.nan, wall=np.nan, steps=0)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    titles = {"asis": "(a) fluid IC as in two_stream.py: velocity seed $u=\\pm(1+0.01\\cos kx)$",
              "matched": "(b) fluid IC matched to NuFI: density seed $\\alpha=10^{-3}$, $T=0.16$"}
    for ax, ic in zip(axs, ("asis", "matched")):
        ax.semilogy(tn, an, color=C.NUFI_COLOR, lw=2.4, label=f"NuFI kinetic  (γ = {fn['gamma']:.4f})",
                    zorder=5)
        for m in C.MODELS_TWOSTREAM:
            fp = C.fluid_trace_path("twostream", m, 0.0, ic)
            if not _have(fp):
                continue
            d = C.load_fluid(fp)
            f = C.growth_fit(d["t"], d["amp"])
            nan_t = float(d["nan_time"])
            rows.append(dict(model=m, ic=ic, **f, nan_time=nan_t, wall=float(d["wall"]),
                             steps=int(d["n_steps"])))
            lab = f"{C.LABEL[m]}  (γ = {f['gamma']:.4f})"
            if np.isfinite(nan_t):
                lab += f", NaN at t≈{nan_t:.1f}"
            ax.semilogy(d["t"], d["amp"], color=C.COLOR[m], lw=1.6, label=lab, zorder=3)
            if np.isfinite(nan_t):
                ax.plot([nan_t], [np.interp(nan_t, d["t"], d["amp"])], marker="X", ms=10,
                        color=C.COLOR[m], mec="#0b0b0b", mew=0.8, zorder=6)
        # kinetic growth-rate guide through NuFI's fit window
        if np.isfinite(fn["t0"]):
            tt = np.linspace(fn["t0"] - 2, fn["t1"] + 2, 50)
            a0 = np.exp(np.interp(0.5 * (fn["t0"] + fn["t1"]), tn, np.log(an)))
            ax.semilogy(tt, 0.5 * a0 * np.exp(gk * (tt - 0.5 * (fn["t0"] + fn["t1"]))), color=C.EXACT_COLOR,
                        ls=":", lw=2, label=f"exact kinetic γ = {gk:.4f} (offset guide)")
        ax.set_title(titles[ic], fontsize=11)
        ax.set_xlabel("t")
        ax.set_xlim(0, 40)
        ax.grid(True, which="major")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    axs[0].set_ylabel(r"$|E_k(t)|$,  $k=0.5$")
    fig.suptitle(r"Two-stream instability ($\pm1$ beams, $T_b=0.16$, $n_b=0.5$, $k=0.5$): NuFI vs 4–7 moment models",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(outfile, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return rows


# ------------------------------------------------------------------
def fmt(v, nd=5):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def main():
    os.makedirs(C.RESULTS, exist_ok=True)
    md = ["# 3–7 moment Landau-fluid models vs NuFI\n",
          "Generated by `validation/scripts/moments/analyze.py`. γ from a linear fit of ln(peak |E_k|) "
          "(same fitter for NuFI and fluid). `legacy` = the original fit in `extract_gamma_nufi.py` / "
          "`run_hp_*.py`.\n",
          f"NuFI electron: {C.NUFI_ELECTRON}, ion: {C.NUFI_ION}, two-stream: {C.NUFI_TWOSTREAM}.  "
          f"Fluid electron: {C.FLUID_ELECTRON}, ion: {C.FLUID_ION}, two-stream: {C.FLUID_TWOSTREAM}.\n"]

    for case, fig in (("electron", "landau_electron_gamma.png"), ("ion", "landau_ion_gamma.png")):
        rows = landau_table(case)
        keys = ["param", "exact", "nufi", "nufi_legacy", "nufi_npk"] + \
               sum([[m, m + "_legacy", m + "_npk"] for m in C.MODELS_LANDAU], [])
        write_csv(rows, os.path.join(C.RESULTS, f"gamma_{case}.csv"), keys)
        plot_landau(rows, case, os.path.join(C.RESULTS, fig))
        plot_fit_check(case, os.path.join(C.RESULTS, f"fit_check_{case}.png"))
        pname = "k" if case == "electron" else "τ"
        md.append(f"\n## {case} damping rate γ\n")
        md.append(f"| {pname} | exact | NuFI | NuFI legacy | " + " | ".join(C.SHORT[m] for m in C.MODELS_LANDAU) + " |")
        md.append("|" + "---|" * (4 + len(C.MODELS_LANDAU)))
        for r in rows:
            md.append(f"| {r['param']:.4g} | {fmt(r['exact'])} | {fmt(r['nufi'])} | {fmt(r['nufi_legacy'])} | " +
                      " | ".join(fmt(r[m]) for m in C.MODELS_LANDAU) + " |")
        # mean |error| vs NuFI in the closure-valid range
        lo = 0.3 if case == "electron" else 0.0
        sel = [r for r in rows if r["param"] >= lo and np.isfinite(r["nufi"])]
        md.append(f"\nMean |γ − γ_NuFI|/|γ_NuFI| over {pname} ≥ {lo}: " + ", ".join(
            f"{C.SHORT[m]} {100*np.nanmean([abs(abs(r[m])-abs(r['nufi']))/abs(r['nufi']) for r in sel]):.1f}%"
            for m in C.MODELS_LANDAU) +
            f"; NuFI vs exact {100*np.nanmean([abs(abs(r['exact'])-abs(r['nufi']))/abs(r['nufi']) for r in sel]):.2f}%\n")

    st = startup("electron", os.path.join(C.RESULTS, "startup_electron.png")) + \
         startup("ion", os.path.join(C.RESULTS, "startup_ion.png"))
    if st:
        write_csv(st, os.path.join(C.RESULTS, "startup.csv"), list(st[0].keys()))
        md.append("\n## Startup: driver IC vs IC matched to NuFI\n")
        md.append("| case | model | IC | max \\|ΔE_k\\|/E_k(0) | first-3-peak error |")
        md.append("|---|---|---|---|---|")
        for r in st:
            md.append(f"| {r['case']} | {C.SHORT[r['model']]} | {r['ic']} | {r['max_dev_over_E0']:.4f} | "
                      f"{r['first3_peak_err_pct']:.2f}% |")

    ts = twostream(os.path.join(C.RESULTS, "twostream_nufi_vs_moments.png"))
    if ts:
        write_csv(ts, os.path.join(C.RESULTS, "twostream.csv"), list(ts[0].keys()))
        md.append(f"\n## Two-stream (exact kinetic γ = {C.exact_twostream().imag:.4f})\n")
        md.append("| model | IC | γ | fit window | pre-growth min at t | saturation |E_k| | t_sat | NaN at t | steps | wall [s] |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in ts:
            name = r["model"] if r["model"] == "NuFI" else C.SHORT[r["model"]]
            md.append(f"| {name} | {r['ic']} | {fmt(r['gamma'], 4)} | [{fmt(r['t0'], 1)}, {fmt(r['t1'], 1)}] | "
                      f"{fmt(r['t_min'], 2)} | {fmt(r['a_sat'], 4)} | {fmt(r['t_sat'], 1)} | "
                      f"{fmt(r['nan_time'], 1)} | {r['steps']} | {fmt(r['wall'], 0)} |")

    with open(os.path.join(C.RESULTS, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
