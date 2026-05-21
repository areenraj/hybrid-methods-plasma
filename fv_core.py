import numpy as np

# R4HP closure constants (Hammett-Perkins 4-moment)
_HP_D1    = 2.0 * np.sqrt(np.pi) / (3.0 * np.pi - 8.0)
_HP_beta1 = (32.0 - 9.0 * np.pi) / (6.0 * np.pi - 16.0)


class Grid:
    def __init__(self, nx, L):
        self.nx = nx
        self.L  = L
        self.dx = L / nx
        self.x  = np.linspace(0.0, L, nx, endpoint=False)
        self.k_rfft     = 2.0 * np.pi * np.fft.rfftfreq(nx, d=self.dx)
        self.abs_k_rfft = np.abs(self.k_rfft)
        self.abs_k_rfft[0] = 1.0

    def spectral_deriv(self, f_hat):
        return np.fft.irfft(1j * self.k_rfft * f_hat, self.nx)

    def poisson_e(self, charge_density, charge_background=1.0):
        """Standard electrostatic solve: dE/dx = charge_difference (periodic, mean(E)=0)."""
        nz = self.k_rfft != 0.0
        c  = charge_background - charge_density
        c = c - np.mean(c)
        c_hat = np.fft.rfft(c)
        E_hat = np.zeros(self.nx // 2 + 1, dtype=complex)
        E_hat[nz] = c_hat[nz] / (1j * self.k_rfft[nz])
        return np.fft.irfft(E_hat, self.nx)

    def poisson_ion(self, charge_density, charge_background=1.0):
        """Ion acoustic Poisson solve: E_hat = -ik * delta_rho_hat."""
        nz = self.k_rfft != 0.0
        rel_charge_density = np.log(charge_density / charge_background)
        phi = np.fft.rfft(rel_charge_density)
        E_hat   = np.zeros(self.nx // 2 + 1, dtype=complex)
        E_hat[nz] = -1j * self.k_rfft[nz] * phi[nz]
        return np.fft.irfft(E_hat, self.nx)


class FluidSolver:
    """
    Finite-volume solver for a single fluid species.

    charge_sign: +1 for ions, -1 for electrons.
    E is passed into rhs() externally — this solver never calls Poisson.
    """

    def __init__(self, grid, model, charge_sign=-1, vt=1.0, gamma=3.0,
                 cs2=1.0, testMode=False, CFL=0.3,
                 rho_floor=1e-10, p_floor=1e-10):
        self.grid        = grid
        self.model       = model
        self.charge_sign = charge_sign
        self.vt          = vt
        self.gamma       = gamma
        self.cs2         = cs2
        self.testMode    = testMode
        self.CFL         = CFL
        self.rho_floor   = rho_floor
        self.p_floor     = p_floor

        # Precompute 4-moment wave speed constants
        disc = gamma**2 * (2*gamma - 1)**2 - 8.0 * gamma
        self._4m_sqrt_disc = np.sqrt(disc)
        self._4m_lam0_coef = np.sqrt((gamma * (2*gamma - 1) + self._4m_sqrt_disc) / 2.0)

    def primitives(self, U):
        rf = self.rho_floor
        pf = self.p_floor
        g  = self.gamma

        if self.model == "pressureless":
            rho, m = U
            rho = np.maximum(rho, rf)
            return rho, m / rho, np.zeros_like(rho), None, None

        if self.model == "isothermal":
            rho, m = U
            rho = np.maximum(rho, rf)
            u   = m / rho
            return rho, u, self.cs2 * rho, None, None

        if self.model == "energy":
            rho, m, energy = U
            rho = np.maximum(rho, rf)
            u   = m / rho
            p   = np.maximum((g - 1.0) * (energy - 0.5 * rho * u**2), pf)
            return rho, u, p, energy, None

        if self.model == "hammett_perkins":
            rho, m, p = U
            rho = np.maximum(rho, rf)
            p   = np.maximum(p, pf)
            u   = m / rho
            return rho, u, p, p / (g - 1.0) + 0.5 * rho * u**2, None

        if self.model == "4moment_hammett_perkins":
            rho, m, p, q = U
            rho = np.maximum(rho, rf)
            p   = np.maximum(p, pf)
            u   = m / rho
            return rho, u, p, p / (g - 1.0) + 0.5 * rho * u**2, q

        raise ValueError(f"Unknown model: {self.model}")

    def physical_flux(self, U):
        rho, u, p, energy, q = self.primitives(U)
        m     = rho * u
        F_rho = m
        F_m   = m * u + p

        if self.model == "energy":
            return np.stack([F_rho, F_m, (energy + p) * u], axis=0)
        if self.model == "hammett_perkins":
            return np.stack([F_rho, F_m, p * u], axis=0)
        if self.model == "4moment_hammett_perkins":
            T = p / rho
            return np.stack([F_rho, F_m, p * u + q, u * q + 3.0 * p * T], axis=0)
        return np.stack([F_rho, F_m], axis=0)

    def _max_wave_speed_4m(self, rho, u, p, q):
        T  = p / np.maximum(rho, self.rho_floor)
        Q  = q / np.maximum(rho, self.rho_floor)
        return np.abs(u) + self._4m_lam0_coef * np.sqrt(T) + np.abs(Q) / (T * self._4m_sqrt_disc)

    def numerical_flux(self, UL, UR):
        FL = self.physical_flux(UL)
        FR = self.physical_flux(UR)
        rhoL, uL, pL, _, qL = self.primitives(UL)
        rhoR, uR, pR, _, qR = self.primitives(UR)
        g = self.gamma

        if self.model == "pressureless":
            alpha = np.maximum(np.abs(uL), np.abs(uR))
        elif self.model == "isothermal":
            c     = np.sqrt(self.cs2)
            alpha = np.maximum(np.abs(uL) + c, np.abs(uR) + c)
        elif self.model == "energy":
            alpha = np.maximum(np.abs(uL) + np.sqrt(g * pL / rhoL),
                               np.abs(uR) + np.sqrt(g * pR / rhoR))
        elif self.model == "hammett_perkins":
            alpha = np.maximum(
                np.maximum(np.abs(uL) + np.sqrt(g * pL / rhoL),
                           np.abs(uR) + np.sqrt(g * pR / rhoR)),
                np.maximum(np.abs(uL), np.abs(uR))
            )
        elif self.model == "4moment_hammett_perkins":
            alpha = np.maximum.reduce([
                np.maximum(np.abs(uL), np.abs(uR)),
                np.maximum(np.abs(uL) + np.sqrt(g * pL / rhoL),
                           np.abs(uR) + np.sqrt(g * pR / rhoR)),
                np.maximum(self._max_wave_speed_4m(rhoL, uL, pL, qL),
                           self._max_wave_speed_4m(rhoR, uR, pR, qR))
            ])
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)

    def _hp_heatflux(self, T, T_hat=None):
        if self.testMode:
            return np.zeros_like(T), np.zeros_like(T)
        gr = self.grid
        if T_hat is None:
            T_hat = np.fft.rfft(T)
        q_hat    = -2.0 * np.sqrt(2.0 / np.pi) * self.vt * (1j * gr.k_rfft / gr.abs_k_rfft) * T_hat
        q_hat[0] = 0.0
        dqdz_hat = 1j * gr.k_rfft * q_hat
        return np.fft.irfft(q_hat, gr.nx), np.fft.irfft(dqdz_hat, gr.nx)

    def _r4hp_r(self, q, T, T_hat=None, q_hat=None):
        if self.testMode:
            return np.zeros_like(T), np.zeros_like(T)
        gr = self.grid
        if q_hat is None:
            q_hat = np.fft.rfft(q)
        if T_hat is None:
            T_hat = np.fft.rfft(T)
        dr_hat = (
            -_HP_D1   * np.sqrt(2.0) * self.vt    * (1j * gr.k_rfft / gr.abs_k_rfft) * q_hat
            + 2.0 * _HP_beta1        * self.vt**2 * T_hat
        )
        dr_hat[0] = 0.0
        return np.fft.irfft(dr_hat, gr.nx), np.fft.irfft(1j * gr.k_rfft * dr_hat, gr.nx)

    def rhs(self, U, E):
        """Semi-discrete RHS. E is the electric field, computed externally."""
        gr    = self.grid
        model = self.model

        UL_plus, UR_plus, UL_minus, UR_minus = _muscl_states(U)
        F_plus  = self.numerical_flux(UL_plus,  UR_plus)
        F_minus = self.numerical_flux(UL_minus, UR_minus)
        dUdt = -(F_plus - F_minus) / gr.dx

        rho = U[0]
        dUdt[1] += self.charge_sign * rho * E

        if model == "energy":
            u = U[1] / np.maximum(rho, self.rho_floor)
            dUdt[2] += self.charge_sign * rho * u * E

        if model == "hammett_perkins":
            rho_s = np.maximum(rho, self.rho_floor)
            u     = U[1] / rho_s
            p     = U[2]
            T     = p / rho_s
            T_hat = np.fft.rfft(T)
            dudx  = gr.spectral_deriv(np.fft.rfft(u))
            dUdt[2] += -2.0 * p * dudx
            _, dqdz = self._hp_heatflux(T, T_hat=T_hat)
            dUdt[2] -= dqdz

        if model == "4moment_hammett_perkins":
            rho_s = np.maximum(rho, self.rho_floor)
            u     = U[1] / rho_s
            p     = U[2]
            q     = U[3]
            T     = p / rho_s
            T_hat = np.fft.rfft(T)
            q_hat = np.fft.rfft(q)
            dudx  = gr.spectral_deriv(np.fft.rfft(u))
            dpdx  = gr.spectral_deriv(np.fft.rfft(p))
            _, ddelrdx = self._r4hp_r(q, T, T_hat=T_hat, q_hat=q_hat)
            dUdt[2] += -2.0 * p * dudx
            dUdt[3] += -3.0 * q * dudx + 3.0 * T * dpdx - ddelrdx

        return dUdt

    def compute_dt(self, U):
        rho, u, p, _, q = self.primitives(U)
        g = self.gamma

        if self.model == "pressureless":
            max_speed = np.max(np.abs(u))
        elif self.model == "isothermal":
            max_speed = np.max(np.abs(u) + np.sqrt(self.cs2))
        elif self.model in ("energy", "hammett_perkins"):
            max_speed = np.max(np.abs(u) + np.sqrt(g * p / rho))
        elif self.model == "4moment_hammett_perkins":
            max_speed = np.max(self._max_wave_speed_4m(rho, u, p, q))
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return self.CFL * self.grid.dx / max(max_speed, 1e-14)

    def kinetic_energy(self, U):
        rho, m = U[0], U[1]
        return np.sum(0.5 * m**2 / np.maximum(rho, 1e-12)) * self.grid.dx

    def internal_energy(self, U):
        if self.model not in ("energy", "hammett_perkins", "4moment_hammett_perkins"):
            return 0.0
        _, _, p, _, _ = self.primitives(U)
        return np.sum(p / (self.gamma - 1.0)) * self.grid.dx


# ============================================================
# MUSCL reconstruction (no grid dependency)
# ============================================================
def _van_leer(a, b):
    ab = a * b
    s  = a + b
    return np.where(ab > 0, 2.0 * ab / np.where(s != 0.0, s, 1.0), 0.0)


def _muscl_states(U):
    dU_r  = np.roll(U, -1, axis=1) - U
    dU_l  = U - np.roll(U,  1, axis=1)
    slope = _van_leer(dU_l, dU_r)
    UL_plus  = U + 0.5 * slope
    UR_plus  = np.roll(U - 0.5 * slope, -1, axis=1)
    UL_minus = np.roll(U + 0.5 * slope,  1, axis=1)
    UR_minus = U - 0.5 * slope
    return UL_plus, UR_plus, UL_minus, UR_minus


# ============================================================
# Time integrators — rhs_fn(U) -> dUdt
# ============================================================
def advance(stepper, rhs_fn, U, dt):
    if stepper == "euler":
        return U + dt * rhs_fn(U)

    if stepper == "ssprk3":
        U1 = U + dt * rhs_fn(U)
        U2 = 0.75 * U + 0.25 * (U1 + dt * rhs_fn(U1))
        return (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * rhs_fn(U2))

    if stepper == "rk4":
        k1 = rhs_fn(U)
        k2 = rhs_fn(U + 0.5 * dt * k1)
        k3 = rhs_fn(U + 0.5 * dt * k2)
        k4 = rhs_fn(U + dt * k3)
        return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    raise ValueError(f"Unknown stepper: {stepper}")
