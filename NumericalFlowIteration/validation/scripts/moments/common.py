"""
Shared paths, model metadata, exact kinetic roots and the damping/growth fitters
used by the 3-7 moment vs NuFI comparison scripts in this folder.
"""

import os
import numpy as np
from scipy.optimize import newton
from scipy.signal import find_peaks
from scipy.special import wofz

HERE      = os.path.dirname(os.path.abspath(__file__))
VALID     = os.path.normpath(os.path.join(HERE, "..", ".."))   # NumericalFlowIteration/validation
NUFI_ROOT = os.path.normpath(os.path.join(VALID, ".."))         # NumericalFlowIteration
REPO      = os.path.normpath(os.path.join(NUFI_ROOT, ".."))     # repo root (fv_core.py, drivers)
RESULTS   = os.environ.get("MOMENTS_RESULTS", os.path.join(VALID, "results", "moments"))  # tables + figures
# raw time traces and NuFI statistics.csv files; bin/* is git-ignored by NumericalFlowIteration/.gitignore
CACHE     = os.environ.get("MOMENTS_CACHE", os.path.join(NUFI_ROOT, "bin", "moments_cache"))

# ------------------------------------------------------------------
# Sweep definitions (same ranges as sweep_electron_k.sh / sweep_ion_tau.sh)
# ------------------------------------------------------------------
ELECTRON_K  = [float(v) for v in np.linspace(0.23, 0.65, 20).round(6)]
ION_K       = 0.0628318530
ION_TAU     = [float(v) for v in np.linspace(0.1, 1.25, 16).round(6)]
DIAG_K      = 0.5          # electron startup diagnostic
DIAG_TAU    = 0.5          # ion startup diagnostic

NUFI_ELECTRON = dict(Nx=64, Nu=256, dt=0.1, T_end=60.0)
NUFI_ION      = dict(Nx=32, Nu=128, dt=0.2, T_end=450.0)
NUFI_TWOSTREAM = dict(Nx=64, Nu=512, dt=0.1, T_end=40.0, u_max=4.0, alpha=0.001,
                      v0=1.0, vth2=0.16, k=0.5)

FLUID_ELECTRON = dict(nx=256, t_end=60.0)
FLUID_ION      = dict(nx=512, t_end=450.0)
FLUID_TWOSTREAM = dict(nx=128, t_end=40.0)

FIT_TMIN_ELECTRON = 5.0
FIT_TMIN_ION      = 40.0

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
MODELS_LANDAU    = ["hammett_perkins", "4moment_hammett_perkins", "5moment_hammett_perkins",
                    "6moment_hammett_perkins", "7moment_hammett_perkins"]
MODELS_TWOSTREAM = MODELS_LANDAU[1:]

LABEL = {
    "hammett_perkins":         r"3-moment (3HP, $R_{3,2}$)",
    "4moment_hammett_perkins": r"4-moment (4HP, $R_{4,3}$)",
    "5moment_hammett_perkins": r"5-moment ($R_{5,3}$)",
    "6moment_hammett_perkins": r"6-moment ($R_{6,4}$)",
    "7moment_hammett_perkins": r"7-moment ($R_{7,5}$)",
}
SHORT = {"hammett_perkins": "3HP", "4moment_hammett_perkins": "4HP",
         "5moment_hammett_perkins": "5M", "6moment_hammett_perkins": "6M",
         "7moment_hammett_perkins": "7M"}
# categorical slots 1-5 of the reference palette, fixed per model (never by rank)
COLOR = {"hammett_perkins": "#2a78d6", "4moment_hammett_perkins": "#eb6834",
         "5moment_hammett_perkins": "#1baf7a", "6moment_hammett_perkins": "#eda100",
         "7moment_hammett_perkins": "#e87ba4"}
MARKER = {"hammett_perkins": "o", "4moment_hammett_perkins": "s",
          "5moment_hammett_perkins": "^", "6moment_hammett_perkins": "D",
          "7moment_hammett_perkins": "v"}
NUFI_COLOR  = "#0b0b0b"
EXACT_COLOR = "#8a8984"
INK2        = "#52514e"


def fluid_trace_path(case, model, param, ic):
    return os.path.join(CACHE, "fluid", case, f"{model}__{param:.6f}__{ic}.npz")


def nufi_stats_path(case, param):
    return os.path.join(CACHE, "nufi", case, f"statistics__{param:.6f}.csv")


# ------------------------------------------------------------------
# Exact kinetic linear roots
# ------------------------------------------------------------------
def _R(z):
    return 1.0 + z * (1j * np.sqrt(np.pi) * wofz(z))


def exact_electron(k):
    """Langmuir root of k^2 + R(w/(sqrt2 k)) = 0 (vt = 1, Maxwellian electrons, fixed ions)."""
    w0 = np.sqrt(1.0 + 3.0 * k * k) - 0.05j
    return complex(newton(lambda w: k * k + _R(w / (np.sqrt(2.0) * k)), w0, tol=1e-13, maxiter=200))


def exact_ion(tau, k=ION_K):
    """Ion-acoustic root with quasineutral Boltzmann electrons (T_e = 1): tau + R(zeta_i) = 0."""
    w0 = k * np.sqrt(1.0 + 3.0 * tau) - 0.3j * k * np.sqrt(tau)
    return complex(newton(lambda w: tau + _R(w / (k * np.sqrt(2.0 * tau))), w0, tol=1e-14, maxiter=200))


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
def load_nufi(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            try:
                rows.append([float(p) for p in line.split(";")])
            except ValueError:
                continue
    return np.array(rows)


def load_fluid(path):
    d = np.load(path, allow_pickle=True)
    return {key: d[key] for key in d.files}


# ------------------------------------------------------------------
# Fitters
# ------------------------------------------------------------------
def refined_peaks(t, a):
    """Local maxima of a(t), refined by a parabola through ln a at the three nearest samples."""
    t = np.asarray(t, float)
    a = np.asarray(a, float)
    idx, _ = find_peaks(a)
    tp, lp = [], []
    for i in idx:
        if i == 0 or i == len(a) - 1 or np.any(a[i - 1:i + 2] <= 0):
            continue
        y0, y1, y2 = np.log(a[i - 1:i + 2])
        h0, h1 = t[i] - t[i - 1], t[i + 1] - t[i]
        # parabola through (-h0,y0),(0,y1),(h1,y2)
        A = ((y2 - y1) / h1 + (y1 - y0) / h0) / (h0 + h1)
        B = (y2 - y1) / h1 - A * h1
        if A < 0:
            s = -B / (2 * A)
            if -h0 <= s <= h1:
                tp.append(t[i] + s)
                lp.append(y1 + B * s + A * s * s)
                continue
        tp.append(t[i])
        lp.append(y1)
    return np.array(tp), np.array(lp)


def envelope_gamma(t, a, t_min, t_max=np.inf, floor_rel=1e-8, rise_tol=0.05):
    """Damping rate from a linear fit of ln(peak |E_k|) vs t.

    Peaks are used from t_min on, stopping at the noise floor (floor_rel * max a),
    at t_max, or at the first peak that rises by more than rise_tol (recurrence/noise).
    Returns (gamma, t_peaks_used, ln_peaks_used).
    """
    a = np.asarray(a, float)
    good = np.isfinite(a)
    t, a = np.asarray(t, float)[good], a[good]
    tp, lp = refined_peaks(t, a)
    floor = np.log(floor_rel * np.max(a))
    keep = []
    for j in range(len(tp)):
        if tp[j] < t_min:
            continue
        if tp[j] > t_max or lp[j] < floor:
            break
        if keep and lp[j] > lp[keep[-1]] + np.log1p(rise_tol):
            break
        keep.append(j)
    if len(keep) < 3:
        return np.nan, tp[keep], lp[keep]
    slope = np.polyfit(tp[keep], lp[keep], 1)[0]
    return float(slope), tp[keep], lp[keep]


def legacy_gamma_fluid(t, a):
    """The original run_hp_*.py fit: ln|E_k| over all samples from the global peak on."""
    i = int(np.nanargmax(a))
    tf, af = t[i:], a[i:]
    m = np.isfinite(af) & (af > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(tf[m], np.log(af[m]), 1)[0])


def legacy_gamma_nufi(data):
    """The original extract_gamma_nufi.py fit (0.5 d ln E_el / dt)."""
    t, E = data[:, 0], data[:, 3]
    m = E > 0
    t, lE = t[m], np.log(E[m])
    i_peak, n = int(np.argmax(E[m])), m.sum()
    sl = slice(max(1, n // 10), i_peak + 1) if i_peak > 0.7 * n else slice(i_peak, n)
    return float(np.polyfit(t[sl], lE[sl], 1)[0] / 2.0)


def growth_fit(t, a, band=(0.35, 0.85)):
    """Linear growth rate of an instability from |E_k(t)|.

    Uses the samples between the pre-growth minimum and the first saturation peak whose
    ln-amplitude lies in the given fractional band of [ln a_min, ln a_sat].
    Returns dict(gamma, t0, t1, a_sat, t_sat, t_min).
    """
    from scipy.ndimage import maximum_filter1d

    t = np.asarray(t, float)
    a = np.asarray(a, float)
    fin = np.isfinite(a) & np.isfinite(t) & (a > 0)
    t, a = t[fin], a[fin]
    # uniform resampling + upper envelope (running max over ~2 time units) removes the
    # near-zero dips of |E_k| caused by the stable, oscillating modes of the seed
    tu = np.arange(t[0], t[-1], 0.02)
    lu = np.interp(tu, t, np.log(a))
    env = maximum_filter1d(lu, size=int(2.0 / 0.02) | 1, mode="nearest")
    # saturation: first local max of |E_k| once it is within a factor 3 of its global max
    big = np.where(lu >= lu.max() - np.log(3.0))[0]
    i = big[0]
    while i + 1 < len(lu) and lu[i + 1] >= lu[i]:
        i += 1
    i_sat = i
    i_min = int(np.argmin(env[:i_sat + 1]))
    lo = env[i_min] + band[0] * (lu[i_sat] - env[i_min])
    hi = env[i_min] + band[1] * (lu[i_sat] - env[i_min])
    sel = np.arange(i_min, i_sat + 1)
    sel = sel[(env[sel] >= lo) & (env[sel] <= hi)]
    if len(sel) < 3:
        g, t0, t1 = np.nan, np.nan, np.nan
    else:
        t0, t1 = float(tu[sel[0]]), float(tu[sel[-1]])
        m = (t >= t0) & (t <= t1)
        g = float(np.polyfit(t[m], np.log(a[m]), 1)[0])
    return dict(gamma=g, t0=t0, t1=t1, a_sat=float(np.exp(lu[i_sat])), t_sat=float(tu[i_sat]),
                t_min=float(tu[i_min]))


def window_fit(t, a, t0, t1):
    t = np.asarray(t, float)
    a = np.asarray(a, float)
    m = (t >= t0) & (t <= t1) & np.isfinite(a) & (a > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(t[m], np.log(a[m]), 1)[0])


def exact_twostream(k=0.5, v0=1.0, T=0.16, n=0.5):
    """Most unstable root of 1 + sum_b n_b/(k^2 T_b) R(zeta_b) = 0 for two drifting Maxwellians."""
    vth = np.sqrt(2.0 * T)

    def D(w):
        return 1.0 + sum(n / (k * k * T) * _R((w - k * u) / (k * vth)) for u in (v0, -v0))

    return complex(newton(D, 0.0 + 0.3j, tol=1e-13, maxiter=200))
