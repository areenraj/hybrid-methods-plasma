import numpy as np
import matplotlib.pyplot as plt
from plasmadisp import electrostatic

MODEL_LABELS = {"hammett_perkins": "3HP Fluid", "4moment_hammett_perkins": "4HP Fluid"}
_npz  = np.load("results/efield_data_electron.npz", allow_pickle=True)
model = MODEL_LABELS.get(str(_npz["model"]), str(_npz["model"]))

data      = np.loadtxt("results/electron_k_results.txt")
if data.ndim == 1:
    data = data[np.newaxis, :]
k_num     = data[:, 0]
gamma_num = data[:, 1]

k_analytic     = np.linspace(k_num.min(), k_num.max(), 300)
gamma_analytic = np.zeros(300)
for i, k in enumerate(k_analytic):
    root = electrostatic.get_roots_to_electrostatic_dispersion(wp_e=1.0, vth_e=1.0, k0=k)
    gamma_analytic[i] = root.imag

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_analytic, -gamma_analytic, ls="-",  lw=2,   color="blue", label="Analytical (Landau)")
ax.plot(k_num,      -gamma_num,      ls="--", lw=1.5, color="red")
ax.scatter(k_num,   -gamma_num,      s=60,            color="red",  label=f"Numerical ({model})", zorder=5)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$-\gamma$ (damping rate)")
ax.set_title("Electron Landau damping rate vs k")
ax.set_yscale("log")
ax.legend()
ax.grid(which="both", ls="--", lw=0.5)

plt.tight_layout()
outfile = "results/electron_k_sweep.png"
plt.savefig(outfile, dpi=500)
plt.show()
print(f"Saved {outfile}")
