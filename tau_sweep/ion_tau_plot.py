import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=float, required=True, help="Wavenumber used in the sweep")
args = parser.parse_args()
k = args.k

MODEL_LABELS = {"hammett_perkins": "3HP Fluid", "4moment_hammett_perkins": "4HP Fluid"}
_npz  = np.load("results/efield_data_ion.npz", allow_pickle=True)
model = MODEL_LABELS.get(str(_npz["model"]), str(_npz["model"]))

data      = np.loadtxt("results/sweep_results.txt")
if data.ndim == 1:
    data = data[np.newaxis, :]
tau_num   = data[:, 0]
gamma_num = data[:, 1]

def find_physical_root(tau, k, cs=1.0):
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

tau_analytic   = np.linspace(tau_num.min(), tau_num.max(), 300)
gamma_analytic = np.zeros(300)
for i, tau in enumerate(tau_analytic):
    root = find_physical_root(tau, k)
    if root is not None:
        gamma_analytic[i] = -root.imag

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(tau_analytic, gamma_analytic, ls="-",  lw=2,   color="blue", label="Analytical (Chapurin Eq 2.49)")
ax.plot(tau_num,     -gamma_num,      ls="--", lw=1.5, color="red")
ax.scatter(tau_num,  -gamma_num,      s=60,            color="red",  label=f"Numerical ({model})", zorder=5)
ax.set_xlabel(r"$\tau = T_{0i}/T_{0e}$")
ax.set_ylabel(r"$-\gamma$ (damping rate)")
ax.set_title(f"Ion acoustic damping rate  k = {k}")
ax.set_yscale("log")
ax.legend()
ax.grid(which="both", ls="--", lw=0.5)

plt.tight_layout()
outfile = f"results/ion_tau_k{k:.2f}.png"
plt.savefig(outfile, dpi=500)
plt.show()
print(f"Saved {outfile}")
