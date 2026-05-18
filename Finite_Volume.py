import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

# Change this depending on ion or electron damping mode.  
# For more information, see the comments in the README
nx = 512
k = 0.3
L = 2 * np.pi / k

testMode=False
ionDamping=False

t_end = 50.0
CFL = 0.3

# This is a bit jury-rigged but its to make sure the mesh doesn't 
# become coarser at different wavenumbers for electron damping
scale_factor = 1.0 if ionDamping else 0.5/k

nx = int(np.ceil(nx*scale_factor))
dx = L / nx
x = np.linspace(0.0, L, nx, endpoint=False)

k_rfft     = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
abs_k_rfft = np.abs(k_rfft)
abs_k_rfft[0] = 1.0

# Choose fluid model:
#   "pressureless"          : 2-moment Euler-Poisson
#   "isothermal"            : 2-moment Euler-Poisson
#   "energy"                : 3-moment Euler-Poisson (adiabatic, gamma=3)
#   "hammett_perkins"       : R3HP closure
#   "4moment_hammett_perkins": R4HP closure
MODEL = "hammett_perkins"

# R4HP parameters
HP_D1     = 2.0 * np.sqrt(np.pi) / (3.0 * np.pi - 8.0)
HP_beta1  = (32.0 - 9.0*np.pi) / (6.0*np.pi - 16.0)

# Choose time integrator:
#   "euler", "ssprk3", "rk4"
TIME_STEPPER = "rk4"

# Live plotting
plot_interval = 50

# Isothermal closure: p = cs2 * rho
cs2 = 1.0

# Ion Electron temperature ratio
tau = 0.5
vt  = np.sqrt(tau) if ionDamping else 1.0  # thermal velocity in cs0 units

# Energy closure: p = (gamma - 1) * (energy - 0.5 rho u^2)
gamma = 3.0

# Background ion density (fixed neutralizing background)
rho_ion = 1.0

# Floors for robustness
rho_floor = 1.0e-10
p_floor = 1.0e-10

# ============================================================
# INITIAL CONDITIONS
# ============================================================
def initial_condition(x, model):
    eps = 0.01
    rho = 1.0 + eps * np.cos(k*x)
    u = np.zeros_like(x)

    if model == "pressureless":
        m = rho * u
        return np.stack([rho, m], axis=0)

    if model == "isothermal":
        m = rho * u
        return np.stack([rho, m], axis=0)

    if model == "energy":
        p = tau * np.ones_like(x) if ionDamping else np.ones_like(x)
        energy = p / (gamma - 1.0) + 0.5 * rho * u**2
        m = rho * u
        return np.stack([rho, m, energy], axis=0)
    if model == "hammett_perkins":
        p = tau * np.ones_like(x) if ionDamping else np.ones_like(x)
        energy = p / (gamma - 1.0) + 0.5 * rho * u**2
        m = rho * u
        return np.stack([rho, m, p], axis=0)
    if model == "4moment_hammett_perkins":
        p = tau * np.ones_like(x) if ionDamping else np.ones_like(x)
        energy = p / (gamma - 1.0) + 0.5 * rho * u**2
        m = rho * u
        q = np.zeros_like(x)
        return np.stack([rho, m, p, q], axis=0)

    raise ValueError("Unknown MODEL")


def spectral_deriv(f_hat):
    return np.fft.irfft(1j * k_rfft * f_hat, nx)


def solve_hp_heatflux(T, T_hat=None):
    
    if testMode:
        return np.zeros_like(T), np.zeros_like(T)

    vt = np.sqrt(tau) if ionDamping else 1.0

    if T_hat is None:
        T_hat = np.fft.rfft(T)

    q_hat    = -2.0 * np.sqrt(2.0 / np.pi) * vt * (1j * k_rfft / abs_k_rfft) * T_hat
    q_hat[0] = 0.0

    dqdz_hat = 1j * k_rfft * q_hat

    return np.fft.irfft(q_hat, nx), np.fft.irfft(dqdz_hat, nx)

def solve_r4hp_r(q, T, T_hat=None, q_hat=None):
    if testMode:
        return np.zeros_like(T), np.zeros_like(T)

    vt = np.sqrt(tau) if ionDamping else 1.0

    if q_hat is None:
        q_hat = np.fft.rfft(q)
    if T_hat is None:
        T_hat = np.fft.rfft(T)

    delta_r_hat = (
        -HP_D1 * np.sqrt(2.0) * vt    * (1j * k_rfft / abs_k_rfft) * q_hat
        + 2.0  * HP_beta1     * vt**2 * T_hat
    )
    delta_r_hat[0] = 0.0

    drdx_hat = 1j * k_rfft * delta_r_hat

    return np.fft.irfft(delta_r_hat, nx), np.fft.irfft(drdx_hat, nx)
    

# ============================================================
# FOURIER-BASED POISSON SOLVER (PERIODIC)
# ============================================================
def compute_electric_field(rho, rho_hat=None):
    """
    Periodic electrostatic solve for electron fluid with fixed ion background.

    We use
        dE/dx = rho_total = rho_ion - rho
    with rho_ion = 1.

    In Fourier space:
        i k E_hat = rho_total_hat
    so
        E_hat = rho_total_hat / (1j * k),   for k != 0

    The k=0 mode is set to zero, corresponding to mean(E) = 0.
    """

    if testMode:
        return np.zeros_like(rho)

    nz = k_rfft != 0.0

    if ionDamping:
        if rho_hat is None:
            rho_hat = np.fft.rfft(rho - 1.0)
        E_hat = np.zeros(nx // 2 + 1, dtype=complex)
        E_hat[nz] = -1j * k_rfft[nz] * rho_hat[nz]
        return np.fft.irfft(E_hat, nx)

    rho_total = rho_ion - rho
    rho_total = rho_total - np.mean(rho_total)
    if rho_hat is None:
        rho_hat = np.fft.rfft(rho_total)
    E_hat = np.zeros(nx // 2 + 1, dtype=complex)
    E_hat[nz] = rho_hat[nz] / (1j * k_rfft[nz])
    return np.fft.irfft(E_hat, nx)

# ============================================================
# MODIFY: PRIMITIVE VARIABLES
# ============================================================
def primitives(U):
    if MODEL == "pressureless":
        rho, m = U
        rho = np.maximum(rho, rho_floor)
        u = m / rho
        p = np.zeros_like(rho)
        return rho, u, p, None, None

    if MODEL == "isothermal":
        rho, m = U
        rho = np.maximum(rho, rho_floor)
        u = m / rho
        p = cs2 * rho
        return rho, u, p, None, None

    if MODEL == "energy":
        rho, m, energy = U
        rho = np.maximum(rho, rho_floor)
        u = m / rho
        kinetic = 0.5 * rho * u**2
        p = (gamma - 1.0) * (energy - kinetic)
        p = np.maximum(p, p_floor)
        return rho, u, p, energy, None
    if MODEL == "hammett_perkins":
        rho, m, p = U
        rho = np.maximum(rho, rho_floor)
        p = np.maximum(p, p_floor)
        u = m / rho
        kinetic = 0.5 * rho * u**2
        energy = p/(gamma - 1.0) + kinetic
        return rho, u, p, energy, None
    if MODEL == "4moment_hammett_perkins":
        rho, m, p, q = U
        rho = np.maximum(rho, rho_floor)
        p = np.maximum(p, p_floor)
        u = m / rho
        kinetic = 0.5 * rho * u**2
        energy = p/(gamma - 1.0) + kinetic
        return rho, u, p, energy, q

    raise ValueError("Unknown MODEL")

# ============================================================
#  PHYSICAL FLUXES
# ============================================================
def physical_flux(U):
    rho, u, p, energy, q = primitives(U)
    m = rho * u

    F_rho = m
    F_m = m * u + p

    if MODEL == "energy":
        F_energy = (energy + p) * u
        return np.stack([F_rho, F_m, F_energy], axis=0)
    
    if MODEL == "hammett_perkins":
        F_p = p * u
        return np.stack([F_rho, F_m, F_p], axis=0)
    
    if MODEL == "4moment_hammett_perkins":
        T = p / rho

        F_p = p * u + q
         
        F_q = u * q + 3 * p * T

        return np.stack([F_rho, F_m, F_p, F_q], axis=0)

    return np.stack([F_rho, F_m], axis=0)

# ============================================================
# 4-MOMENT WAVE SPEED (This was written by GPT - I assumed that there must have been 
#                      issues with the numerical fluxes for the new moment but I didn't
#                      know how to handle it. It improves agreement with damping quite a bit)
# ============================================================

# Flux Jacobian co-moving characteristic polynomial: λ^4 - γ(2γ-1)T λ^2 + 2γT^2 = 0
# disc = γ^2(2γ-1)^2 - 8γ  →  largest root λ0 = sqrt(T*(γ(2γ-1)+sqrt(disc))/2)
# First-order q correction (conservative upper bound): +|q/ρ| / (T * sqrt(disc))

_4M_DISC      = gamma**2 * (2*gamma - 1)**2 - 8.0*gamma
_4M_SQRT_DISC = np.sqrt(_4M_DISC)
_4M_LAM0_COEF = np.sqrt((gamma*(2*gamma - 1) + _4M_SQRT_DISC) / 2.0)

def max_wave_speed_4moment(rho, u, p, q):
    rho_s = np.maximum(rho, rho_floor)
    T     = p / rho_s
    Q     = q / rho_s
    lam0       = _4M_LAM0_COEF * np.sqrt(T)
    correction = np.abs(Q) / (T * _4M_SQRT_DISC)
    return np.abs(u) + lam0 + correction

# ============================================================
# NUMERICAL FLUXES (RUSANOV)
# ============================================================
def numerical_flux(UL, UR):
    FL = physical_flux(UL)
    FR = physical_flux(UR)

    rhoL, uL, pL, _, qL = primitives(UL)
    rhoR, uR, pR, _, qR = primitives(UR)

    if MODEL == "pressureless":
        alpha = np.maximum(np.abs(uL), np.abs(uR))

    elif MODEL == "isothermal":
        c = np.sqrt(cs2)
        alpha = np.maximum(np.abs(uL) + c, np.abs(uR) + c)

    elif MODEL == "energy":
        cL = np.sqrt(gamma * pL / rhoL)
        cR = np.sqrt(gamma * pR / rhoR)
        alpha = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    elif MODEL == "hammett_perkins":
        cL = np.sqrt(gamma * pL / rhoL)
        cR = np.sqrt(gamma * pR / rhoR)
        alpha_acoustic  = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)
        alpha_advective = np.maximum(np.abs(uL),       np.abs(uR))
        # rho and m rows get acoustic speed; p row gets advective speed only
        # because F_p = p*u has eigenvalue u, not u±c
        alpha = np.stack([alpha_acoustic, alpha_acoustic, alpha_acoustic], axis=0)

    elif MODEL == "4moment_hammett_perkins":
        cL = np.sqrt(gamma * pL / rhoL)
        cR = np.sqrt(gamma * pR / rhoR)

        alpha_advective = np.maximum(np.abs(uL),       np.abs(uR))
        alpha_acoustic  = np.maximum(np.abs(uL) + cL,  np.abs(uR) + cR)
        alpha_4moment   = np.maximum(
            max_wave_speed_4moment(rhoL, uL, pL, qL),
            max_wave_speed_4moment(rhoR, uR, pR, qR))
        # F_rho=m (advective), F_m=mu+p (acoustic),
        # F_p=pu+q (advective), F_q=uq+3pT (full 4-moment speed)
        alpha = np.stack([alpha_acoustic, alpha_acoustic, alpha_acoustic, alpha_4moment], axis=0)

    else:
        raise ValueError("Unknown MODEL")

    return 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)

# ============================================================
# MUSCL RECONSTRUCTION (van Leer limiter)
# ============================================================
def _van_leer(a, b):
    ab = a * b
    s = a + b
    return np.where(ab > 0, 2.0 * ab / np.where(s != 0.0, s, 1.0), 0.0)

def _muscl_states(U):
    dU_r = np.roll(U, -1, axis=1) - U
    dU_l = U - np.roll(U,  1, axis=1)
    slope = _van_leer(dU_l, dU_r)
    UL_plus  = U + 0.5 * slope
    UR_plus  = np.roll(U - 0.5 * slope, -1, axis=1)
    UL_minus = np.roll(U + 0.5 * slope,  1, axis=1)
    UR_minus = U - 0.5 * slope
    return UL_plus, UR_plus, UL_minus, UR_minus

# ============================================================
# SEMI-DISCRETE RHS
# ============================================================
def rhs(U):
    rho    = U[0]
    dn_hat = np.fft.rfft(rho - 1.0) if ionDamping else None
    Efield = compute_electric_field(rho, rho_hat=dn_hat)

    UL_plus, UR_plus, UL_minus, UR_minus = _muscl_states(U)
    F_plus  = numerical_flux(UL_plus,  UR_plus)
    F_minus = numerical_flux(UL_minus, UR_minus)

    dUdt = -(F_plus - F_minus) / dx

    # Momentum source (electrons)
    dUdt[1] += +rho * Efield if ionDamping else -rho * Efield

    if MODEL == "energy":
        rho_safe = np.maximum(rho, rho_floor)
        u = U[1] / rho_safe
        dUdt[2] += -rho * u * Efield

    if MODEL == "hammett_perkins":
        rho_safe = np.maximum(rho, rho_floor)
        u = U[1] / rho_safe
        p = U[2]

        T     = p / rho_safe
        T_hat = np.fft.rfft(T)
        dudx  = spectral_deriv(np.fft.rfft(u))

        dUdt[2] += -2.0 * p * dudx

        q, dqdz = solve_hp_heatflux(T, T_hat=T_hat)
        dUdt[2] += -dqdz

    if MODEL == "4moment_hammett_perkins":

        rho_safe = np.maximum(rho, rho_floor)
        u = U[1] / rho_safe
        p = U[2]
        q = U[3]

        T     = p / rho_safe
        T_hat = np.fft.rfft(T)
        q_hat = np.fft.rfft(q)
        dudx  = spectral_deriv(np.fft.rfft(u))
        dpdx  = spectral_deriv(np.fft.rfft(p))

        _, ddelrdx = solve_r4hp_r(q, T, T_hat=T_hat, q_hat=q_hat)

        dUdt[2] += - 2.0 * p * dudx 

        dUdt[3] +=  -3.0 * q * dudx
        dUdt[3] +=   3.0 * T * dpdx
        dUdt[3] += - ddelrdx

    return dUdt

# ============================================================
# TIME STEP
# ============================================================
def compute_dt(U):
    rho, u, p, _, _ = primitives(U)

    if MODEL == "pressureless":
        max_speed = np.max(np.abs(u))

    elif MODEL == "isothermal":
        max_speed = np.max(np.abs(u) + np.sqrt(cs2))

    elif MODEL in ("energy", "hammett_perkins"):
        c = np.sqrt(gamma * p / rho)
        max_speed = np.max(np.abs(u) + c)

    elif MODEL == "4moment_hammett_perkins":
        _, _, _, _, q = primitives(U)
        max_speed = np.max(max_wave_speed_4moment(rho, u, p, q))

    else:
        raise ValueError("Unknown MODEL")

    return CFL * dx / max(max_speed, 1e-14)

# ============================================================
# TIME INTEGRATORS
# ============================================================
def step_euler(U, dt):
    return U + dt * rhs(U)

def step_ssprk3(U, dt):
    U1 = U + dt * rhs(U)
    U2 = 0.75 * U + 0.25 * (U1 + dt * rhs(U1))
    U3 = (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * rhs(U2))
    return U3

def step_rk4(U, dt):
    k1 = rhs(U)
    k2 = rhs(U + 0.5 * dt * k1)
    k3 = rhs(U + 0.5 * dt * k2)
    k4 = rhs(U + dt * k3)

    return U + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

def advance(U, dt):
    if TIME_STEPPER == "euler":
        return step_euler(U, dt)
    if TIME_STEPPER == "ssprk3":
        return step_ssprk3(U, dt)
    if TIME_STEPPER == "rk4":
        return step_rk4(U, dt)
    raise ValueError("Unknown TIME_STEPPER")

# ============================================================
# ENERGIES
# ============================================================
def compute_kinetic_energy(U):
    rho = U[0]
    m = U[1]

    rho_safe = np.maximum(rho, 1e-12)
    return np.sum(0.5 * m**2 / rho_safe) * dx

def compute_internal_energy(U):
    if MODEL not in ("energy", "hammett_perkins", "4moment_hammett_perkins"):
        return 0.0

    rho, u, p, _, _ = primitives(U)
    return np.sum(p / (gamma - 1.0)) * dx

# ============================================================
# LIVE PLOTTING SETUP
# ============================================================
def setup_plot(U):
    plt.ion()

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 2)

    ax_rho = fig.add_subplot(gs[0, 0])
    ax_u   = fig.add_subplot(gs[0, 1])
    ax_p   = fig.add_subplot(gs[1, 0])
    ax_E   = fig.add_subplot(gs[1, 1])

    ax_energy = fig.add_subplot(gs[2, :])

    rho, u, p, _, _ = primitives(U)

    Efield = compute_electric_field(rho)

    line_rho, = ax_rho.plot(x, rho)
    line_u,   = ax_u.plot(x, u)
    line_p,   = ax_p.plot(x, p)
    line_E,   = ax_E.plot(x, Efield)

    E_energy = 0.5 * np.sum(Efield**2) * dx
    K_energy = compute_kinetic_energy(U)
    I_energy = compute_internal_energy(U)
    T_energy = E_energy + K_energy + I_energy

    time_data = [0.0]
    E_data = [E_energy]
    K_data = [K_energy]
    I_data = [I_energy]
    T_data = [T_energy]

    line_E_energy, = ax_energy.semilogy(time_data, E_data, label="Electric")
    line_K_energy, = ax_energy.semilogy(time_data, K_data, label="Kinetic")
    line_I_energy, = ax_energy.semilogy(time_data, I_data, label="Internal", marker='o')
    line_T_energy, = ax_energy.semilogy(time_data, T_data, label="Total")

    ax_rho.set_ylabel("rho_e")
    ax_u.set_ylabel("u")
    ax_p.set_ylabel("p")
    ax_E.set_ylabel("E(x)")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_xlabel("t")
    ax_energy.legend()

    for ax in [ax_rho, ax_u, ax_p, ax_E, ax_energy]:
        ax.grid()

    title = fig.suptitle(f"{MODEL}, {TIME_STEPPER}, t = 0.000")
    plt.tight_layout()

    return fig, (ax_rho, ax_u, ax_p, ax_E, ax_energy), \
           (line_rho, line_u, line_p, line_E,
            line_E_energy, line_K_energy,
            line_I_energy, line_T_energy), \
           (time_data, E_data, K_data, I_data, T_data), title


def update_plot(U, Efield, axes, lines, history, title, t):

    ax_rho, ax_u, ax_p, ax_E, ax_energy = axes
    line_rho, line_u, line_p, line_E, \
    line_E_energy, line_K_energy, \
    line_I_energy, line_T_energy = lines

    time_data, E_data, K_data, I_data, T_data = history

    rho, u, p, _, _ = primitives(U)

    line_rho.set_ydata(rho)
    line_u.set_ydata(u)
    line_p.set_ydata(p)
    line_E.set_ydata(Efield)

    E_energy = 0.5 * np.sum(Efield**2) * dx
    K_energy = compute_kinetic_energy(U)
    I_energy = compute_internal_energy(U)
    T_energy = E_energy + K_energy + I_energy

    time_data.append(t)
    E_data.append(E_energy)
    K_data.append(K_energy)
    I_data.append(I_energy)
    T_data.append(T_energy)

    line_E_energy.set_data(time_data, E_data)
    line_K_energy.set_data(time_data, K_data)
    line_I_energy.set_data(time_data, I_data)
    line_T_energy.set_data(time_data, T_data)

    for ax in [ax_rho, ax_u, ax_p, ax_E]:
        ax.relim()
        ax.autoscale_view()

    ax_energy.relim()
    ax_energy.autoscale_view()

    title.set_text(f"{MODEL}, {TIME_STEPPER}, t = {t:.3f}")
    plt.pause(0.001)

# ============================================================
# MAIN
# ============================================================
def main():
    if ionDamping and MODEL not in ("hammett_perkins", "4moment_hammett_perkins"):
        raise ValueError("Ion damping requires a kinetic closure (hammett_perkins or 4moment_hammett_perkins)")

    U = initial_condition(x, MODEL)
    t = 0.0
    step_count = 0

    do_plot = plot_interval < 10**8
    if do_plot:
        fig, axes, lines, history, title = setup_plot(U)

    efield_time = []
    efield_amplitude = []
    rho_mode_amplitude = []
    E_kmode_amplitude = []

    # Fourier bin index for the initial perturbation wavenumber.
    # Domain L = 2π/k contains exactly one full wavelength, so it sits in bin 1.
    _k_bin = 1

    while t < t_end:

        dt = compute_dt(U)

        if TIME_STEPPER == "rk4":
            dt *= 0.5

        if t + dt > t_end:
            dt = t_end - t

        U = advance(U, dt)

        t += dt
        step_count += 1

        Efield = compute_electric_field(U[0])

        efield_time.append(t)
        efield_amplitude.append(np.max(np.abs(Efield)))

        rho_hat = np.fft.fft(U[0] - 1.0)
        rho_mode_amplitude.append(np.abs(rho_hat[_k_bin]) * 2.0 / nx)

        E_hat = np.fft.fft(Efield)
        E_kmode_amplitude.append(E_hat[_k_bin])

        if do_plot and step_count % plot_interval == 0:
            update_plot(U, Efield, axes, lines, history, title, t)

    if do_plot:
        Efield = compute_electric_field(U[0])
        update_plot(U, Efield, axes, lines, history, title, t)
        plt.ioff()
        plt.show()

    E_kmode = np.array(E_kmode_amplitude)
    outfile = "results/efield_data_ion.npz" if ionDamping else "results/efield_data_electron.npz"
    extra = {"tau": tau} if ionDamping else {}
    np.savez(
        outfile,
        t=np.array(efield_time),
        E_amp=np.array(efield_amplitude),
        rho_mode_amp=np.array(rho_mode_amplitude),
        model=MODEL,
        k=k,
        nx=np.array([nx]),
        L=np.array([L]),
        E_kmode_real=E_kmode.real,
        E_kmode_imag=E_kmode.imag,
        **extra,
    )

if __name__ == "__main__":
    main()

