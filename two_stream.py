import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from fv_core import Grid, FluidSolver

# ============================================================
# PARAMETERS
# ============================================================

nx  = 128
v_b = 1.0    # beam velocity (each beam at ±v_b)
k   = 0.5    # perturbation wavenumber
L   = 2 * np.pi / k

# Models: "pressureless", "isothermal", "energy", "hammett_perkins", "4moment_hammett_perkins"
MODEL        = "4moment_hammett_perkins"
TIME_STEPPER = "rk4"

t_end          = 40.0
t_record_start = 0.0
CFL            = 0.05

plot_interval = 50
plot_mode     = "v2"   # "v1": phase space / density / E / pressure
               #        "v2": per-beam (rho,u) / vel diff / E
               
save_movie    = False #ffmpeg required!
movie_file    = "results/two_stream.mp4"
movie_fps     = 20

gamma     = 3.0
cs2       = 1.0
rho_floor = 1.0e-10
p_floor   = 1.0e-10


# ============================================================
# INITIAL CONDITIONS
# Two equal-density cold beams at ±v_b with a shared density
# perturbation. Each beam carries half the background density.
# ============================================================
def initial_conditions(x_arr, model, k_val, v_b_val, eps=0.01):
    rho1 = 0.5 * np.ones_like(x_arr)
    rho2 = 0.5 * np.ones_like(x_arr)

    u1 = +v_b_val + eps*np.cos(k_val*x_arr)
    u2 = -v_b_val - eps*np.cos(k_val*x_arr)

    m1 = rho1*u1
    m2 = rho2*u2

    if model in ("pressureless", "isothermal"):
        return np.stack([rho1, m1], axis=0), np.stack([rho2, m2], axis=0)

    p = 0.01 * np.ones_like(x_arr)

    if model == "energy":
        energy1 = p / (gamma - 1.0) + 0.5 * rho1 * u1**2
        energy2 = p / (gamma - 1.0) + 0.5 * rho2 * u2**2
        return np.stack([rho1, m1, energy1], axis=0), np.stack([rho2, m2, energy2], axis=0)
    if model == "hammett_perkins":
        return np.stack([rho1, m1, p], axis=0), np.stack([rho2, m2, p], axis=0)
    if model == "4moment_hammett_perkins":
        q = np.zeros_like(x_arr)
        return np.stack([rho1, m1, p, q], axis=0), np.stack([rho2, m2, p, q], axis=0)

    raise ValueError(f"Model '{model}' not supported for two-stream")


# ============================================================
# PLOTTING
# ============================================================
def setup_plot(U1, U2, solver1, solver2, grid):
    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(3, 2)

    ax_ps  = fig.add_subplot(gs[0, 0])  # phase space
    ax_dn  = fig.add_subplot(gs[0, 1])  # density perturbation
    ax_E   = fig.add_subplot(gs[1, 0])  # electric field
    ax_p   = fig.add_subplot(gs[1, 1])  # total pressure
    ax_en  = fig.add_subplot(gs[2, :])  # energy

    rho1, u1, p1, _, _ = solver1.primitives(U1)
    rho2, u2, p2, _, _ = solver2.primitives(U2)
    E = grid.poisson_e(U1[0] + U2[0])

    line_u1,  = ax_ps.plot(grid.x, u1, label="beam 1")
    line_u2,  = ax_ps.plot(grid.x, u2, label="beam 2")
    line_dn,  = ax_dn.plot(grid.x, rho1 + rho2 - 1.0)
    line_E,   = ax_E.plot(grid.x, E)
    line_p,   = ax_p.plot(grid.x, p1 + p2)

    Ee = 0.5 * np.sum(E**2) * grid.dx
    Ke = solver1.kinetic_energy(U1) + solver2.kinetic_energy(U2)
    Ie = solver1.internal_energy(U1) + solver2.internal_energy(U2)
    Te = Ee + Ke + Ie

    time_data = [0.0]
    Ee_data, Ke_data, Ie_data, Te_data = [Ee], [Ke], [Ie], [Te]

    line_Ee, = ax_en.semilogy(time_data, Ee_data, label="Electric")
    line_Ke, = ax_en.semilogy(time_data, Ke_data, label="Kinetic")
    line_Ie, = ax_en.semilogy(time_data, Ie_data, label="Internal", marker='o')
    line_Te, = ax_en.semilogy(time_data, Te_data, label="Total")

    ax_ps.set_ylabel("u");             ax_ps.legend(); ax_ps.grid()
    ax_dn.set_ylabel("rho1+rho2-1");   ax_dn.grid()
    ax_E.set_ylabel("E(x)");           ax_E.grid()
    ax_p.set_ylabel("p1+p2");          ax_p.grid()
    ax_en.set_ylabel("Energy");        ax_en.set_xlabel("t")
    ax_en.legend();                    ax_en.grid()

    title = fig.suptitle(f"Two-stream  {MODEL}, t = 0.000")
    plt.tight_layout()

    return (fig,
            (ax_ps, ax_dn, ax_E, ax_p, ax_en),
            (line_u1, line_u2, line_dn, line_E, line_p,
             line_Ee, line_Ke, line_Ie, line_Te),
            (time_data, Ee_data, Ke_data, Ie_data, Te_data))


def update_plot(U1, U2, E, solver1, solver2, grid, axes, lines, history, title, t):
    ax_ps, ax_dn, ax_E, ax_p, ax_en = axes
    line_u1, line_u2, line_dn, line_E_l, line_p, \
    line_Ee, line_Ke, line_Ie, line_Te = lines
    time_data, Ee_data, Ke_data, Ie_data, Te_data = history

    rho1, u1, p1, _, _ = solver1.primitives(U1)
    rho2, u2, p2, _, _ = solver2.primitives(U2)

    line_u1.set_ydata(u1);             line_u2.set_ydata(u2)
    line_dn.set_ydata(rho1 + rho2 - 1.0)
    line_E_l.set_ydata(E)
    line_p.set_ydata(p1 + p2)

    Ee = 0.5 * np.sum(E**2) * grid.dx
    Ke = solver1.kinetic_energy(U1) + solver2.kinetic_energy(U2)
    Ie = solver1.internal_energy(U1) + solver2.internal_energy(U2)
    Te = Ee + Ke + Ie

    time_data.append(t)
    Ee_data.append(Ee); Ke_data.append(Ke)
    Ie_data.append(Ie); Te_data.append(Te)

    line_Ee.set_data(time_data, Ee_data); line_Ke.set_data(time_data, Ke_data)
    line_Ie.set_data(time_data, Ie_data); line_Te.set_data(time_data, Te_data)

    for ax in axes:
        ax.relim(); ax.autoscale_view()

    title.set_text(f"Two-stream  {MODEL}, t = {t:.3f}")
    plt.pause(0.001)


def _e_to_phi(E, grid):
    nz = grid.k_rfft != 0.0
    E_hat = np.fft.rfft(E)
    phi_hat = np.zeros_like(E_hat)
    phi_hat[nz] = E_hat[nz] / (-1j * grid.k_rfft[nz])
    return np.fft.irfft(phi_hat, grid.nx)


def setup_plot_v2(U1, U2, solver1, solver2, grid):
    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(3, 2)

    ax_b1  = fig.add_subplot(gs[0, 0])  # beam 1: rho1, u1
    ax_b2  = fig.add_subplot(gs[0, 1])  # beam 2: rho2, u2
    ax_phi = fig.add_subplot(gs[1, 0])  # electric potential
    ax_dv  = fig.add_subplot(gs[1, 1])  # velocity difference
    ax_en = fig.add_subplot(gs[2, :])  # energy

    rho1, u1, _, _, _ = solver1.primitives(U1)
    rho2, u2, _, _, _ = solver2.primitives(U2)
    E = grid.poisson_e(U1[0] + U2[0])

    line_rho1, = ax_b1.plot(rho1, u1, '.')
    line_rho2, = ax_b2.plot(rho2, u2, '.')

    phi = _e_to_phi(E, grid)
    line_phi, = ax_phi.plot(grid.x, phi)
    line_dv,  = ax_dv.plot(grid.x, u1 - u2)

    Ee = 0.5 * np.sum(E**2) * grid.dx
    Ke = solver1.kinetic_energy(U1) + solver2.kinetic_energy(U2)
    Ie = solver1.internal_energy(U1) + solver2.internal_energy(U2)
    Te = Ee + Ke + Ie

    time_data = [0.0]
    Ee_data, Ke_data, Ie_data, Te_data = [Ee], [Ke], [Ie], [Te]

    line_Ee, = ax_en.semilogy(time_data, Ee_data, label="Electric")
    line_Ke, = ax_en.semilogy(time_data, Ke_data, label="Kinetic")
    line_Ie, = ax_en.semilogy(time_data, Ie_data, label="Internal", marker='o')
    line_Te, = ax_en.semilogy(time_data, Te_data, label="Total")

    ax_b1.set_xlabel("rho1");  ax_b1.set_ylabel("u1");  ax_b1.grid()
    ax_b2.set_xlabel("rho2");  ax_b2.set_ylabel("u2");  ax_b2.grid()
    ax_phi.set_ylabel("phi(x)"); ax_phi.grid()
    ax_dv.set_ylabel("u1-u2");   ax_dv.grid()
    ax_en.set_ylabel("Energy"); ax_en.set_xlabel("t")
    ax_en.legend(); ax_en.grid()

    title = fig.suptitle(f"Two-stream  {MODEL}, t = 0.000")
    plt.tight_layout()

    return (fig,
            (ax_b1, ax_b2, ax_phi, ax_dv, ax_en),
            (line_rho1, line_rho2, line_phi, line_dv,
             line_Ee, line_Ke, line_Ie, line_Te),
            (time_data, Ee_data, Ke_data, Ie_data, Te_data))


def update_plot_v2(U1, U2, E, solver1, solver2, grid, axes, lines, history, title, t):
    ax_b1, ax_b2, ax_phi, ax_dv, ax_en = axes
    line_rho1, line_rho2, line_phi, line_dv, \
    line_Ee, line_Ke, line_Ie, line_Te = lines
    time_data, Ee_data, Ke_data, Ie_data, Te_data = history

    rho1, u1, _, _, _ = solver1.primitives(U1)
    rho2, u2, _, _, _ = solver2.primitives(U2)

    line_rho1.set_data(rho1, u1)
    line_rho2.set_data(rho2, u2)
    line_phi.set_ydata(_e_to_phi(E, grid))
    line_dv.set_ydata(u1 - u2)

    Ee = 0.5 * np.sum(E**2) * grid.dx
    Ke = solver1.kinetic_energy(U1) + solver2.kinetic_energy(U2)
    Ie = solver1.internal_energy(U1) + solver2.internal_energy(U2)
    Te = Ee + Ke + Ie

    time_data.append(t)
    Ee_data.append(Ee); Ke_data.append(Ke)
    Ie_data.append(Ie); Te_data.append(Te)

    line_Ee.set_data(time_data, Ee_data); line_Ke.set_data(time_data, Ke_data)
    line_Ie.set_data(time_data, Ie_data); line_Te.set_data(time_data, Te_data)

    for ax in axes:
        ax.relim(); ax.autoscale_view()

    title.set_text(f"Two-stream  {MODEL}, t = {t:.3f}")
    plt.pause(0.001)


# ============================================================
# MAIN
# ============================================================
def main():
    grid    = Grid(nx, L)
    solver1 = FluidSolver(grid, MODEL, charge_sign=-1, gamma=gamma, cs2=cs2,
                          CFL=CFL, rho_floor=rho_floor, p_floor=p_floor)
    solver2 = FluidSolver(grid, MODEL, charge_sign=-1, gamma=gamma, cs2=cs2,
                          CFL=CFL, rho_floor=rho_floor, p_floor=p_floor)

    U1, U2 = initial_conditions(grid.x, MODEL, k, v_b)

    # ---- coupled RHS: E field depends on both species ----
    def rhs_coupled(u1, u2):
        E = grid.poisson_e(u1[0] + u2[0])
        return solver1.rhs(u1, E), solver2.rhs(u2, E)

    def advance_coupled(u1, u2, dt):
        if TIME_STEPPER == "euler":
            d1, d2 = rhs_coupled(u1, u2)
            return u1 + dt * d1, u2 + dt * d2
        if TIME_STEPPER == "ssprk3":
            d1, d2 = rhs_coupled(u1, u2)
            v1 = u1 + dt*d1;  v2 = u2 + dt*d2
            d1, d2 = rhs_coupled(v1, v2)
            w1 = 0.75*u1 + 0.25*(v1 + dt*d1);  w2 = 0.75*u2 + 0.25*(v2 + dt*d2)
            d1, d2 = rhs_coupled(w1, w2)
            return (u1/3.0 + 2.0/3.0*(w1 + dt*d1),
                    u2/3.0 + 2.0/3.0*(w2 + dt*d2))
        if TIME_STEPPER == "rk4":
            k1a, k1b = rhs_coupled(u1, u2)
            k2a, k2b = rhs_coupled(u1 + 0.5*dt*k1a, u2 + 0.5*dt*k1b)
            k3a, k3b = rhs_coupled(u1 + 0.5*dt*k2a, u2 + 0.5*dt*k2b)
            k4a, k4b = rhs_coupled(u1 + dt*k3a,     u2 + dt*k3b)
            return (u1 + (dt/6.0)*(k1a + 2.0*k2a + 2.0*k3a + k4a),
                    u2 + (dt/6.0)*(k1b + 2.0*k2b + 2.0*k3b + k4b))
        raise ValueError(f"Unknown TIME_STEPPER: {TIME_STEPPER}")

    t = 0.0
    step_count = 0

    do_plot = plot_interval < 10**8
    writer  = None
    if do_plot:
        if plot_mode == "v2":
            fig, axes, lines, history = setup_plot_v2(U1, U2, solver1, solver2, grid)
        else:
            fig, axes, lines, history = setup_plot(U1, U2, solver1, solver2, grid)
        title = fig.texts[0] if fig.texts else fig.suptitle("")
        _update = update_plot_v2 if plot_mode == "v2" else update_plot
        if save_movie:
            writer = FFMpegWriter(fps=movie_fps)
            writer.setup(fig, movie_file, dpi=100)
            writer.grab_frame()

    efield_time      = []
    efield_amplitude = []
    E_kmode_amplitude = []
    _k_bin = 1

    while t < t_end:
        dt = min(solver1.compute_dt(U1), solver2.compute_dt(U2))
        if t + dt > t_end:
            dt = t_end - t

        U1, U2 = advance_coupled(U1, U2, dt)
        t += dt
        step_count += 1

        E = grid.poisson_e(U1[0] + U2[0])

        if t >= t_record_start:
            efield_time.append(t)
            efield_amplitude.append(np.max(np.abs(E)))
            E_hat = np.fft.fft(E)
            E_kmode_amplitude.append(E_hat[_k_bin])

        if do_plot and step_count % plot_interval == 0:
            _update(U1, U2, E, solver1, solver2, grid, axes, lines, history, title, t)
            if writer:
                writer.grab_frame()

    if do_plot:
        E = grid.poisson_e(U1[0] + U2[0])
        _update(U1, U2, E, solver1, solver2, grid, axes, lines, history, title, t)
        if writer:
            writer.grab_frame()
            writer.finish()
            print(f"Movie saved to {movie_file}")
        plt.ioff()
        plt.show()

    E_kmode = np.array(E_kmode_amplitude)
    np.savez(
        "results/efield_data_twostream.npz",
        t            = np.array(efield_time),
        E_amp        = np.array(efield_amplitude),
        model        = MODEL,
        k            = k,
        v_b          = v_b,
        nx           = np.array([nx]),
        L            = np.array([L]),
        E_kmode_real = E_kmode.real,
        E_kmode_imag = E_kmode.imag,
    )


if __name__ == "__main__":
    main()
