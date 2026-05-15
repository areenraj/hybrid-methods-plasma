import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# CONFIGURATION
#   "E_amp"        : electric field amplitude  → physics + numerical damping
#                    also triggers Landau damping comparison
#   "rho_mode_amp" : density Fourier mode amplitude → numerical damping only
# ============================================================
FIELD_KEY = "rho_mode_amp"

FIELD_LABELS = {
    "E_amp":        "|E|_max",
    "rho_mode_amp": "|δρ_k| (density mode)",
}
field_label = FIELD_LABELS.get(FIELD_KEY, FIELD_KEY)

# ============================================================
# LOAD DATA
# ============================================================
data       = np.load("efield_data.npz", allow_pickle=True)
t          = data["t"]
field_amp  = data[FIELD_KEY]
model      = str(data["model"])
k          = float(data["k"])

# ============================================================
# FIT EXPONENTIAL DECAY
# field_amp(t) ~ A * exp(gamma * t)
# Skip initial transient by starting from the peak.
# ============================================================
def fit_decay(t_arr, amp_arr):
    i_peak = np.argmax(amp_arr)
    t_f = t_arr[i_peak:]
    a_f = amp_arr[i_peak:]

    mask = a_f > 0
    t_f, a_f = t_f[mask], a_f[mask]

    def linear(t, gamma, log_A):
        return gamma * t + log_A

    popt, pcov = curve_fit(linear, t_f, np.log(a_f),
                           p0=[-0.01, np.log(a_f[0])])
    gamma_fit, log_A_fit = popt
    perr = np.sqrt(np.diag(pcov))
    return gamma_fit, np.exp(log_A_fit), perr[0], t_f, a_f, i_peak


gamma_fit, A_fit, gamma_err, t_fit, a_fit, i_peak = fit_decay(t, field_amp)

print(f"Model     : {model}")
print(f"k         : {k:.4f}")
print(f"Field     : {field_label}")
print(f"gamma_fit = {gamma_fit:.6f}  +/- {gamma_err:.6f}")

# ============================================================
# LANDAU DAMPING COMPARISON — only for electric field
# ============================================================
gamma_landau = None
omega_landau = None
if FIELD_KEY == "E_amp":
    from plasmadisp import electrostatic

    omega_p = 1.0
    vth     = 1.0

    root = electrostatic.get_roots_to_electrostatic_dispersion(
        wp_e=omega_p, vth_e=vth, k0=k
    )
    omega_landau  = root.real
    gamma_landau  = root.imag

    print(f"omega_r   (Landau) : {omega_landau:.6f}")
    print(f"gamma     (Landau) : {gamma_landau:.6f}")
    print(f"Relative error     : "
          f"{abs(gamma_fit - gamma_landau)/abs(gamma_landau)*100:.2f}%")

# ============================================================
# PLOT
# ============================================================
n_panels = 2 if gamma_landau is not None else 1
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
if n_panels == 1:
    axes = [axes]

# Panel 1: amplitude time series + fit
ax = axes[0]
ax.semilogy(t, field_amp, label=f"Numerical {field_label}", lw=1.5)

t_overlay = np.linspace(t_fit[0], t_fit[-1], 300)
ax.semilogy(t_overlay, A_fit * np.exp(gamma_fit * t_overlay),
            "--", label=f"Fit: γ = {gamma_fit:.4f}", lw=2)

ax.axvline(t[i_peak], color="gray", ls=":", label="Fit start (peak)")
ax.set_xlabel("t")
ax.set_ylabel(field_label)
ax.set_title(f"{model}, k={k:.3f}")
ax.legend()
ax.grid()

# Panel 2: bar chart comparison (only when E field)
if gamma_landau is not None:
    ax2 = axes[1]
    labels = ["Analytical\n(Landau)", f"Numerical\n({model})"]
    values = [gamma_landau, gamma_fit]
    colors = ["steelblue", "tomato"]

    bars = ax2.bar(labels, values, color=colors, width=0.4)
    ax2.axhline(0, color="k", lw=0.8)

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 val - 0.002 * np.sign(val),
                 f"{val:.5f}", ha="center", va="top", fontsize=10)

    ax2.set_ylabel("Damping rate γ")
    ax2.set_title("Damping rate comparison")
    ax2.grid(axis="y")

plt.tight_layout()
plt.savefig("damping_comparison.png", dpi=150)
plt.show()
print("Plot saved to damping_comparison.png")