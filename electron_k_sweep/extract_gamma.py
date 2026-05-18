import numpy as np

data    = np.load("results/efield_data_electron.npz", allow_pickle=True)
t       = data["t"]
nx      = int(data["nx"].item())
E_kmode = data["E_kmode_real"] + 1j * data["E_kmode_imag"]
amp     = np.abs(E_kmode) * 2.0 / nx

i_peak   = np.argmax(amp)
t_f      = t[i_peak:]
a_f      = amp[i_peak:]
mask     = a_f > 0
t_f, a_f = t_f[mask], a_f[mask]

coeffs    = np.polyfit(t_f, np.log(a_f), 1)
gamma_fit = coeffs[0]

print(f"{gamma_fit:.8f}")
