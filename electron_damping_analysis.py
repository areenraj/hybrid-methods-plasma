import numpy as np
import matplotlib.pyplot as plt
from plasmadisp import electrostatic

data    = np.load("results/efield_data_electron.npz", allow_pickle=True)
t       = data["t"]
MODEL_LABELS = {"hammett_perkins": "3HP Fluid", "4moment_hammett_perkins": "4HP Fluid"}
model   = MODEL_LABELS.get(str(data["model"]), str(data["model"]))
k       = float(data["k"])

if "E_kmode_real" not in data:
    raise KeyError("E_kmode_real not found. Re-run Finite_Volume.py to regenerate the npz.")

E_kmode = data["E_kmode_real"] + 1j * data["E_kmode_imag"]
amp     = np.abs(E_kmode) * 2.0 / int(data["nx"])

i_peak  = np.argmax(amp)
t_post  = t[i_peak:]
a_post  = amp[i_peak:]
mask    = a_post > 0
t_m     = t_post[mask]
a_m     = a_post[mask]

coeffs    = np.polyfit(t_m, np.log(a_m), 1)
gamma_num = coeffs[0]
A_fit     = np.exp(coeffs[1])

root         = electrostatic.get_roots_to_electrostatic_dispersion(wp_e=1.0, vth_e=1.0, k0=k)
gamma_landau = root.imag

print(f"gamma (numerical) : {gamma_num:.6f}")
print(f"gamma (Landau)    : {gamma_landau:.6f}")
print(f"relative error    : {abs(gamma_num - gamma_landau) / abs(gamma_landau) * 100:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.semilogy(t, amp, lw=1.5, label="|Ê(k,t)|")
t_ov = np.linspace(t_m[0], t_m[-1], 400)
ax.semilogy(t_ov, A_fit * np.exp(gamma_num * t_ov), "--", lw=2, label=f"γ = {gamma_num:.4f}")
ax.axvline(t[i_peak], color="gray", ls=":", label="fit start")
ax.set_xlabel("t")
ax.set_ylabel("|Ê(k)|")
ax.set_title(f"{model}, k={k:.3f}")
ax.legend()
ax.grid()

ax2 = axes[1]
labels = ["Analytical\n(Landau)", f"Numerical\n({model})"]
values = [gamma_landau, gamma_num]
colors = ["steelblue", "tomato"]
bars   = ax2.bar(labels, values, color=colors, width=0.4)
ax2.axhline(0, color="k", lw=0.8)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val - 0.002 * np.sign(val),
             f"{val:.5f}", ha="center", va="top", fontsize=10)
ax2.set_ylabel("Damping rate γ")
ax2.set_title("Damping rate comparison")
ax2.grid(axis="y")

plt.tight_layout()
plt.savefig("results/electron_damping_analysis.png", dpi=150)
plt.show()
print("Plot saved to electron_damping_analysis.png")
