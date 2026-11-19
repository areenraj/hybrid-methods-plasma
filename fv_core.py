import numpy as np

# R4HP closure constants (Hammett-Perkins 4-moment)
_HP_D1    = 2.0 * np.sqrt(np.pi) / (3.0 * np.pi - 8.0)
_HP_beta1 = (32.0 - 9.0 * np.pi) / (6.0 * np.pi - 16.0)

# R53 closure constants for the 5th moment X5 (Hunana et al. 2019, eq. 464)
_R53_den = 9.0 * np.pi**2 - 69.0 * np.pi + 128.0
_R53_Ar  = (104.0 - 33.0 * np.pi) * np.sqrt(np.pi) / (2.0 * _R53_den)
_R53_Aq  = (81.0 * np.pi - 256.0) / (2.0 * _R53_den)
_R53_AT  = 3.0 * (160.0 - 51.0 * np.pi) * np.sqrt(np.pi) / (4.0 * _R53_den)
_R53_Au  = (135.0 * np.pi**2 - 750.0 * np.pi + 1024.0) / (2.0 * _R53_den)

# R64 closure constants for X6_tilde = X6 - 15 p^3 / rho^2 (Hunana et al. 2019, eq. 475)
_R64_den = 801.0 * np.pi**2 - 5124.0 * np.pi + 8192.0
_R64_e5  = 3.0 * (180.0 * np.pi**2 - 1197.0 * np.pi + 1984.0) * np.sqrt(np.pi) / _R64_den
_R64_er  = 3.0 * (675.0 * np.pi**2 - 4728.0 * np.pi + 8192.0) / (2.0 * _R64_den)
_R64_eq  = 3.0 * (285.0 * np.pi - 896.0) * np.sqrt(np.pi) / (2.0 * _R64_den)
_R64_eT  = 3.0 * (7065.0 * np.pi**2 - 43056.0 * np.pi + 65536.0) / (4.0 * _R64_den)
_R64_eu  = 9.0 * (450.0 * np.pi**2 - 2799.0 * np.pi + 4352.0) * np.sqrt(np.pi) / _R64_den

# R75 constants (Hunana et al. 2019, eq. 478); used as a static X7 closure, see _hi_closure()
_R75_den = 10800.0 * np.pi**3 - 120915.0 * np.pi**2 + 440160.0 * np.pi - 524288.0
_R75_a6  = 18.0 * (1545.0 * np.pi**2 - 9743.0 * np.pi + 15360.0) * np.sqrt(np.pi) / _R75_den
_R75_b5  = 3.0 * (52425.0 * np.pi**2 - 331584.0 * np.pi + 524288.0) / (2.0 * _R75_den)
_R75_br  = 3.0 * (7875.0 * np.pi**2 - 50490.0 * np.pi + 80896.0) * np.sqrt(np.pi) / _R75_den
_R75_bq  = 3.0 * (162000.0 * np.pi**3 - 1758825.0 * np.pi**2 + 6263040.0 * np.pi - 7340032.0) / (4.0 * _R75_den)
_R75_bT  = 27.0 * (15825.0 * np.pi**2 - 99260.0 * np.pi + 155648.0) * np.sqrt(np.pi) / (2.0 * _R75_den)
_R75_bu  = 3.0 * (189000.0 * np.pi**3 - 1612215.0 * np.pi**2 + 4534656.0 * np.pi - 4194304.0) / (2.0 * _R75_den)

# models evolving the central moments W_2..W_{N-1} and closing W_N
_HI_MODELS = {"5moment_hammett_perkins": 5, "6moment_hammett_perkins": 6, "7moment_hammett_perkins": 7}


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

        # N-moment max wave speed / sqrt(T): local (non-Hilbert) linear system about a Maxwellian
        if model in _HI_MODELS:
            self._hi_coef = _hi_speed_coef(_HI_MODELS[model])

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

        if self.model == "4moment_hammett_perkins" or self.model in _HI_MODELS:
            rho, m, p, q = U[:4]
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
        if self.model in _HI_MODELS:
            # flux of W_l is u W_l + W_{l+1}; W_N is replaced by its local Grad part,
            # the remainder of W_N is closed in rhs()
            N = _HI_MODELS[self.model]
            W = [None, None] + [U[l] for l in range(2, N)]
            W.append(_grad_part(N, rho, p, W))
            return np.stack([F_rho, F_m] + [u * W[l] + W[l + 1] for l in range(2, N)], axis=0)
        return np.stack([F_rho, F_m], axis=0)

    def _max_wave_speed_4m(self, rho, u, p, q):
        T  = p / np.maximum(rho, self.rho_floor)
        Q  = q / np.maximum(rho, self.rho_floor)
        return np.abs(u) + self._4m_lam0_coef * np.sqrt(T) + np.abs(Q) / (T * self._4m_sqrt_disc)

    def _max_wave_speed_hi(self, rho, u, p):
        T = p / np.maximum(rho, self.rho_floor)
        return np.abs(u) + self._hi_coef * np.sqrt(T)

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
        elif self.model in _HI_MODELS:
            alpha = np.maximum(self._max_wave_speed_hi(rhoL, uL, pL),
                               self._max_wave_speed_hi(rhoR, uR, pR))
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)

    def _hp_heatflux(self, n0, T, T_hat=None):
        if self.testMode:
            return np.zeros_like(T), np.zeros_like(T)
        gr = self.grid
        if T_hat is None:
            T_hat = np.fft.rfft(T)
        q_hat    = -2.0 * np.sqrt(2.0 / np.pi) * n0 * self.vt * (1j * gr.k_rfft / gr.abs_k_rfft) * T_hat
        q_hat[0] = 0.0
        dqdz_hat = 1j * gr.k_rfft * q_hat
        return np.fft.irfft(q_hat, gr.nx), np.fft.irfft(dqdz_hat, gr.nx)

    def _r4hp_r(self, n0, q, T, T_hat=None, q_hat=None):
        if self.testMode:
            return np.zeros_like(T), np.zeros_like(T)
        gr = self.grid
        if q_hat is None:
            q_hat = np.fft.rfft(q)
        if T_hat is None:
            T_hat = np.fft.rfft(T)
        dr_hat = (
            -_HP_D1   * np.sqrt(2.0)      * self.vt    * (1j * gr.k_rfft / gr.abs_k_rfft) * q_hat
            + 2.0 * _HP_beta1        * n0 * self.vt**2 * T_hat
        )
        dr_hat[0] = 0.0
        return np.fft.irfft(dr_hat, gr.nx), np.fft.irfft(1j * gr.k_rfft * dr_hat, gr.nx)

    def _hi_closure(self, N, n0, rho, u, U):
        """Linear closure for W_N minus its Grad part (_grad_part), Hunana et al. 2019, v_th = sqrt(2) vt.

        N=5: R53, eq. 464.  N=6: R64, eq. 475.  N=7: R75, static X7 obtained from the dynamic
        eq. 478 and the linear X6_tilde equation  d_t X6_tilde + d_x X7 - 45 T0^2 d_x q = 0.
        The Grad part's linearisation is subtracted so the linear response is unchanged.
        """
        if self.testMode:
            return np.zeros_like(rho), np.zeros_like(rho)
        gr  = self.grid
        vth = np.sqrt(2.0) * self.vt
        T0  = self.vt**2
        H   = 1j * gr.k_rfft / gr.abs_k_rfft
        F   = np.fft.rfft
        p, q, r = U[2], U[3], U[4]
        T  = p / rho
        rt = r - 3.0 * p**2 / rho

        if N == 5:   # Grad part 10 T q
            w_hat = (- _R53_Ar * vth * H * F(rt)
                     - (_R53_Aq * vth**2 + 10.0 * T0) * F(q)
                     - _R53_AT * vth**3 * n0 * H * F(T)
                     - _R53_Au * vth**2 * n0 * T0 * F(u))
        elif N == 6:   # Grad part 15 p^3/rho^2 + 15 T r_tilde
            X5 = U[5]
            w_hat = (- _R64_e5 * vth * H * F(X5)
                     + (_R64_er * vth**2 - 15.0 * T0) * F(rt)
                     - _R64_eq * vth**3 * H * F(q)
                     - _R64_eT * vth**4 * n0 * F(T)
                     - _R64_eu * vth**3 * n0 * T0 * H * F(u))
        else:   # N == 7, Grad part 21 T X5 - 105 T^2 q
            X5  = U[5]
            X6t = U[6] - 15.0 * p**3 / rho**2
            w_hat = (_R75_a6 * vth * H * F(X6t)
                     - (_R75_b5 * vth**2 + 21.0 * T0) * F(X5)
                     - _R75_br * vth**3 * H * F(rt)
                     - (_R75_bq * vth**4 - 150.0 * T0**2) * F(q)
                     + _R75_bT * vth**5 * n0 * H * F(T)
                     - _R75_bu * vth**4 * n0 * T0 * F(u))
        w_hat[0] = 0.0
        return np.fft.irfft(w_hat, gr.nx), np.fft.irfft(1j * gr.k_rfft * w_hat, gr.nx)

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
            _, dqdz = self._hp_heatflux(np.mean(rho), T, T_hat=T_hat)
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
            _, ddelrdx = self._r4hp_r(np.mean(rho), q, T, T_hat=T_hat, q_hat=q_hat)
            dUdt[2] += -2.0 * p * dudx
            dUdt[3] += -3.0 * q * dudx + 3.0 * T * dpdx - ddelrdx

        if model in _HI_MODELS:
            # moment hierarchy, Hammett et al. 1992 eq. 11:
            # d_t W_l + d_x(u W_l + W_{l+1}) = -l W_l u_x + l (W_{l-1}/rho) p_x,  closed at W_N
            N     = _HI_MODELS[model]
            rho_s = np.maximum(rho, self.rho_floor)
            u     = U[1] / rho_s
            dudx  = gr.spectral_deriv(np.fft.rfft(u))
            dpdx  = gr.spectral_deriv(np.fft.rfft(U[2]))
            for l in range(2, N):
                dUdt[l] += -l * U[l] * dudx
                if l >= 3:
                    dUdt[l] += l * (U[l - 1] / rho_s) * dpdx
            _, dwdx = self._hi_closure(N, np.mean(rho), rho_s, u, U)
            dUdt[N - 1] -= dwdx

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
        elif self.model in _HI_MODELS:
            max_speed = np.max(self._max_wave_speed_hi(rho, u, p))
        else:
            raise ValueError(f"Unknown model: {self.model}")

        return self.CFL * self.grid.dx / max(max_speed, 1e-14)

    def kinetic_energy(self, U):
        rho, m = U[0], U[1]
        return np.sum(0.5 * m**2 / np.maximum(rho, 1e-12)) * self.grid.dx

    def internal_energy(self, U):
        if self.model not in ("energy", "hammett_perkins", "4moment_hammett_perkins") \
                and self.model not in _HI_MODELS:
            return 0.0
        _, _, p, _, _ = self.primitives(U)
        return np.sum(p / (self.gamma - 1.0)) * self.grid.dx


# ============================================================
# N-moment helpers (Grad part of the closed moment, wave-speed bound)
# ============================================================
def _grad_part(N, rho, p, W):
    """Local part of W_N for a Maxwellian plus Hermite (Grad) corrections, T = p/rho.

    N=5: 10 T q.   N=6: 15 T r - 30 p^3/rho^2  (= 15 p^3/rho^2 + 15 T r_tilde).
    N=7: 21 T X5 - 105 T^2 q.
    """
    T = p / rho
    if N == 5:
        return 10.0 * T * W[3]
    if N == 6:
        return 15.0 * T * W[4] - 30.0 * p**3 / rho**2
    return 21.0 * T * W[5] - 105.0 * T**2 * W[3]


def _hi_speed_coef(N):
    """Max |eigenvalue| of the local (non-Hilbert) linearised N-moment system, in units of sqrt(T).

    Variables (n, u, p, q, r, X5, X6) about n0 = T0 = 1, where W_l0 = (l-1)!! for even l.
    The last row adds the non-Hilbert part of the full linearised closure for W_N.
    """
    W0 = [1.0, 0.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0]
    M = np.zeros((N, N))
    M[0, 1] = 1.0
    M[1, 2] = 1.0
    for l in range(2, N):
        M[l, 1] = (l + 1) * W0[l]
        M[l, 2] = -l * W0[l - 1]
        if l + 1 < N:
            M[l, l + 1] = 1.0
    c = np.zeros(N)
    vth2 = 2.0
    if N == 5:
        c[3] = -_R53_Aq * vth2
        c[1] = -_R53_Au * vth2
    elif N == 6:   # 45 p - 30 n + er vth^2 (r - 6p + 3n) - eT vth^4 (p - n)
        c[0] = -30.0 + 3.0 * _R64_er * vth2 + _R64_eT * vth2**2
        c[2] = 45.0 - 6.0 * _R64_er * vth2 - _R64_eT * vth2**2
        c[4] = _R64_er * vth2
    else:
        c[5] = -_R75_b5 * vth2
        c[3] = -(_R75_bq * vth2**2 - 45.0)
        c[1] = -_R75_bu * vth2**2
    M[N - 1] += c
    return np.max(np.abs(np.linalg.eigvals(M)))


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
