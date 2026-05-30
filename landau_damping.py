import numpy as np
import matplotlib.pyplot as plt

from fv_core import Grid, FluidSolver, advance

# ============================================================
# PARAMETERS
# These module-level variables can be mutated by sweep scripts
# before calling main() — e.g.:
#   import landau_damping as sim
#   sim.tau = 0.3; sim.ionDamping = True; sim.main()
# ============================================================

nx = 512
L  = 100
k  = 2 * np.pi / L


testMode   = False
ionDamping = True

t_end          = 450.0
t_record_start = 0.0
CFL            = 0.2

# Models: "pressureless", "isothermal", "energy",
#         "hammett_perkins", "4moment_hammett_perkins"
MODEL        = "4moment_hammett_perkins"
TIME_STEPPER = "rk4"

plot_interval = 1e7

cs2   = 1.0
tau   = 0.5
gamma = 3.0

background_charge   = 1.0
rho_floor = 1.0e-10
p_floor   = 1.0e-10

scale_factor = 1.0 if ionDamping else 0.5 / k
nx           = int(np.ceil(nx * scale_factor))


# ============================================================
# INITIAL CONDITIONS
# ============================================================
def initial_condition(x_arr, model, k_val, tau_val, ion_damping, gamma_val):
    eps = 0.01
    rho = 1.0 + eps * np.cos(k_val * x_arr)
    u   = np.zeros_like(x_arr)
    m   = rho * u

    if model in ("pressureless", "isothermal"):
        return np.stack([rho, m], axis=0)

    p0 = tau_val if ion_damping else 1.0
    p  = p0 * np.ones_like(x_arr)

    if model == "energy":
        energy = p / (gamma_val - 1.0) + 0.5 * rho * u**2
        return np.stack([rho, m, energy], axis=0)
    if model == "hammett_perkins":
        return np.stack([rho, m, p], axis=0)
    if model == "4moment_hammett_perkins":
        q = np.zeros_like(x_arr)
        return np.stack([rho, m, p, q], axis=0)

    raise ValueError(f"Unknown model: {model}")


# ============================================================
# PLOTTING
# ============================================================
def setup_plot(U, solver, grid):
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    gs  = fig.add_gridspec(3, 2)

    ax_rho    = fig.add_subplot(gs[0, 0])
    ax_u      = fig.add_subplot(gs[0, 1])
    ax_p      = fig.add_subplot(gs[1, 0])
    ax_E      = fig.add_subplot(gs[1, 1])
    ax_energy = fig.add_subplot(gs[2, :])

    rho, u, p, _, _ = solver.primitives(U)
    E = _compute_E(grid, U[0], background_charge)

    line_rho, = ax_rho.plot(grid.x, rho)
    line_u,   = ax_u.plot(grid.x, u)
    line_p,   = ax_p.plot(grid.x, p)
    line_E,   = ax_E.plot(grid.x, E)

    Ee = 0.5 * np.sum(E**2) * grid.dx
    Ke = solver.kinetic_energy(U)
    Ie = solver.internal_energy(U)
    Te = Ee + Ke + Ie

    time_data = [0.0]
    E_data, K_data, I_data, T_data = [Ee], [Ke], [Ie], [Te]

    line_Ee, = ax_energy.semilogy(time_data, E_data, label="Electric")
    line_Ke, = ax_energy.semilogy(time_data, K_data, label="Kinetic")
    line_Ie, = ax_energy.semilogy(time_data, I_data, label="Internal", marker='o')
    line_Te, = ax_energy.semilogy(time_data, T_data, label="Total")

    ax_rho.set_ylabel("rho_e"); ax_u.set_ylabel("u")
    ax_p.set_ylabel("p");       ax_E.set_ylabel("E(x)")
    ax_energy.set_ylabel("Energy"); ax_energy.set_xlabel("t")
    ax_energy.legend()
    for ax in [ax_rho, ax_u, ax_p, ax_E, ax_energy]:
        ax.grid()

    title = fig.suptitle(f"{MODEL}, {TIME_STEPPER}, t = 0.000")
    plt.tight_layout()

    return (fig,
            (ax_rho, ax_u, ax_p, ax_E, ax_energy),
            (line_rho, line_u, line_p, line_E, line_Ee, line_Ke, line_Ie, line_Te),
            (time_data, E_data, K_data, I_data, T_data),
            title)


def update_plot(U, E, solver, grid, axes, lines, history, title, t):
    ax_rho, ax_u, ax_p, ax_E, ax_energy = axes
    line_rho, line_u, line_p, line_E_l, line_Ee, line_Ke, line_Ie, line_Te = lines
    time_data, E_data, K_data, I_data, T_data = history

    rho, u, p, _, _ = solver.primitives(U)
    line_rho.set_ydata(rho); line_u.set_ydata(u)
    line_p.set_ydata(p);     line_E_l.set_ydata(E)

    Ee = 0.5 * np.sum(E**2) * grid.dx
    Ke = solver.kinetic_energy(U)
    Ie = solver.internal_energy(U)
    Te = Ee + Ke + Ie

    time_data.append(t); E_data.append(Ee)
    K_data.append(Ke);   I_data.append(Ie); T_data.append(Te)

    line_Ee.set_data(time_data, E_data); line_Ke.set_data(time_data, K_data)
    line_Ie.set_data(time_data, I_data); line_Te.set_data(time_data, T_data)

    for ax in [ax_rho, ax_u, ax_p, ax_E]:
        ax.relim(); ax.autoscale_view()
    ax_energy.relim(); ax_energy.autoscale_view()

    title.set_text(f"{MODEL}, {TIME_STEPPER}, t = {t:.3f}")
    plt.pause(0.001)


# ============================================================
# HELPERS
# ============================================================
def _compute_E(grid, rho, background_charge):
    if ionDamping:
        return grid.poisson_ion(rho, background_charge)
    return grid.poisson_e(rho, background_charge)


# ============================================================
# MAIN
# ============================================================
def main():
    if ionDamping and MODEL not in ("hammett_perkins", "4moment_hammett_perkins"):
        raise ValueError("Ion damping requires hammett_perkins or 4moment_hammett_perkins")

    # Derive runtime values from current module globals
    vt_val      = np.sqrt(tau) if ionDamping else 1.0
    charge      = +1 if ionDamping else -1
    grid        = Grid(nx, L)
    solver      = FluidSolver(grid, MODEL, charge_sign=charge, vt=vt_val,
                              gamma=gamma, cs2=cs2, testMode=testMode,
                              CFL=CFL, rho_floor=rho_floor, p_floor=p_floor)

    U = initial_condition(grid.x, MODEL, k, tau, ionDamping, gamma)
    t = 0.0
    step_count = 0

    do_plot = plot_interval < 10**8
    if do_plot:
        fig, axes, lines, history, title = setup_plot(U, solver, grid)

    efield_time       = []
    efield_amplitude  = []
    rho_mode_amplitude = []
    E_kmode_amplitude  = []
    _k_bin = 1

    def rhs_fn(u):
        rho_u = u[0]
        E     = np.zeros(grid.nx) if testMode else _compute_E(grid, rho_u, background_charge)
        return solver.rhs(u, E)

    while t < t_end:
        dt = solver.compute_dt(U)

        if t + dt > t_end:
            dt = t_end - t

        U = advance(TIME_STEPPER, rhs_fn, U, dt)
        t += dt
        step_count += 1

        # for recording and plotting
        E = np.zeros(grid.nx) if testMode else _compute_E(grid, U[0], background_charge)

        if t >= t_record_start:
            efield_time.append(t)
            efield_amplitude.append(np.max(np.abs(E)))

            rho_hat = np.fft.fft(U[0] - 1.0)
            rho_mode_amplitude.append(np.abs(rho_hat[_k_bin]) * 2.0 / grid.nx)

            E_hat = np.fft.fft(E)
            E_kmode_amplitude.append(E_hat[_k_bin])

        if do_plot and step_count % plot_interval == 0:
            update_plot(U, E, solver, grid, axes, lines, history, title, t)

    if do_plot:
        E = np.zeros(grid.nx) if testMode else _compute_E(grid, U[0], background_charge)
        update_plot(U, E, solver, grid, axes, lines, history, title, t)
        plt.ioff()
        plt.show()

    E_kmode = np.array(E_kmode_amplitude)
    outfile = "results/efield_data_ion.npz" if ionDamping else "results/efield_data_electron.npz"
    extra   = {"tau": tau} if ionDamping else {}
    np.savez(
        outfile,
        t              = np.array(efield_time),
        E_amp          = np.array(efield_amplitude),
        rho_mode_amp   = np.array(rho_mode_amplitude),
        model          = MODEL,
        k              = k,
        nx             = np.array([grid.nx]),
        L              = np.array([grid.L]),
        E_kmode_real   = E_kmode.real,
        E_kmode_imag   = E_kmode.imag,
        **extra,
    )


if __name__ == "__main__":
    main()
