# Landau-fluid moment models in `fv_core.py`

Notes on the 3-, 4- and 5-moment Hammett–Perkins (HP) models: the equations, the closures, how the code implements them, and the changes made on 2026-11-19.

**References**
- [HP90] Hammett & Perkins, PRL 64, 3019 (1990)
- [HDP92] Hammett, Dorland & Perkins, Phys. Fluids B 4, 2052 (1992)
- [H19] Hunana et al., *An introductory guide to fluid models with anisotropic temperatures, Part 2*, arXiv:1901.09360. Useful pages: 17–20 (kinetic moments), 40–45 (closures), 49–51 (real space and the $q$ equation), 54–56 (5-pole closures), 69–70 (closures for $X_5$).

---

## 1. Notation and units

The code sets $m = 1$, so $\rho = n$. Each species has a background density $n_0 = \langle\rho\rangle$ and temperature $T_0$.

| Symbol | Code | Meaning |
|---|---|---|
| $v_t = \sqrt{T_0/m}$ | `self.vt` | thermal speed as defined in HP90 |
| $v_{th} = \sqrt{2T_0/m} = \sqrt2\,v_t$ | `np.sqrt(2)*self.vt` | thermal speed as defined in H19 |
| $T_0 = v_t^2$ | `self.vt**2` | background temperature |
| $n_0$ | `np.mean(rho)` | background density |
| $\mathcal H \leftrightarrow i\,\mathrm{sgn}(k)$ | `1j*k/|k|` | Hilbert transform (H19 sign convention) |
| $\zeta = \omega / (\lvert k\rvert v_{th})$ | | normalised phase velocity |

The central moments are

$$
p = m\!\int (v-u)^2 f\,dv,\quad
q = m\!\int (v-u)^3 f\,dv,\quad
r = m\!\int (v-u)^4 f\,dv,\quad
X_5 = m\!\int (v-u)^5 f\,dv ,
$$

with $T = p/\rho$ and $\tilde r = r - 3p^2/\rho$. The $\tilde r$ part is what is left after removing the Maxwellian value of $r$.

Every closure below is linear, applied in Fourier space, and has the $k=0$ mode removed. As a result only fluctuations about the box mean enter the closures.

---

## 2. General moment hierarchy

Take $W_l = m\int (v-u)^l f\,dv$. The 1D Vlasov equation gives, exactly ([HDP92] Eq. 11):

$$
\partial_t W_l + \partial_x (u W_l) = -\,l\,W_l\,\partial_x u + l\,\frac{W_{l-1}}{\rho}\,\partial_x p - \partial_x W_{l+1}.
$$

For $l \ge 2$ the electric field drops out, because it cancels against the momentum equation. Only the momentum equation contains $E$. An $N$-moment model evolves $W_0,\dots,W_{N-1}$ (with $W_1$ replaced by $\rho u$) and needs a closure for $W_N$.

---

## 3. 3-moment model (`hammett_perkins`), closure $R_{3,2}$

Evolved: $(\rho,\ \rho u,\ p)$. Closure for $q$ ([HP90] Eq. 7, [H19] Eq. 276):

$$
\hat q_k = -\frac{2}{\sqrt\pi}\, n_0 v_{th}\, i\,\mathrm{sgn}(k)\,\hat T_k
       = -2\sqrt{\tfrac{2}{\pi}}\; n_0 v_t\, i\,\mathrm{sgn}(k)\,\hat T_k .
$$

The code implements this in `_hp_heatflux`.

---

## 4. 4-moment model (`4moment_hammett_perkins`), closure $R_{4,3}$

Evolved: $(\rho,\ \rho u,\ p,\ q)$. The $q$ equation ([H19] Eq. 340) is

$$
\partial_t q + \partial_x\!\left(uq + 3pT\right) = -3q\,\partial_x u + 3T\,\partial_x p - \partial_x \tilde r .
$$

Closure for $\tilde r$ ([HP90] between Eqs. 10 and 11, [H19] Eqs. 292 and 324):

$$
\hat{\tilde r}_k = -\frac{2\sqrt\pi}{3\pi-8}\, v_{th}\, i\,\mathrm{sgn}(k)\,\hat q_k
                 + \frac{32-9\pi}{2(3\pi-8)}\, n_0 v_{th}^2\, \hat T_k .
$$

In the HP90 constants used by the code, $D_1 = \dfrac{2\sqrt\pi}{3\pi-8}$ and $\beta_1 = \dfrac{32-9\pi}{6\pi-16}$, this reads

$$
\hat{\tilde r}_k = -D_1\sqrt2\, v_t\, i\,\mathrm{sgn}(k)\,\hat q_k + 2\beta_1\, n_0 v_t^2\, \hat T_k .
$$

### Bug fixed: extra $n_0$

The code had `-_HP_D1 * sqrt(2) * n0 * vt * ...`, which multiplies the $\hat q$ term by $n_0$. Neither paper has that factor ($q$ already carries the density). The bug had no effect when $n_0 = 1$ (the electron and ion runs). It halved the damping of $q$ in the two-stream run, where $n_0 = 0.5$ per beam. The factor has been removed. A numerical check confirms that the linearised 4-moment code reproduces $R_{4,3}(\zeta)$ to machine precision.

---

## 5. 5-moment model (`5moment_hammett_perkins`), closure $R_{5,3}$ (new)

### 5.1 Which closure

Among the 5-pole Padé approximants, H19 marks as "reliable" ($\checkmark$ in Eq. 300) the closures whose fluid dispersion relation equals the kinetic one with $R \to R_{n,n'}$. Of those, $R_{5,3}$ is the most precise. It has accuracy $o(\zeta^4)$ for small $\zeta$ and $o(\zeta^{-5})$ for large $\zeta$, and it is the "R53" curve in H19 Fig. 9. The tabulated maximum error of $R_{5,3}$ against the exact $R(\zeta)$ is about 1.5–1.8%.

### 5.2 Equations

Evolved: $U = (\rho,\ \rho u,\ p,\ q,\ r)$, from §2 with $l = 0\ldots4$:

$$
\begin{aligned}
&\partial_t \rho + \partial_x(\rho u) = 0,\\
&\partial_t(\rho u) + \partial_x(\rho u^2 + p) = q_s\,\rho E,\\
&\partial_t p + \partial_x(pu + q) = -2p\,\partial_x u,\\
&\partial_t q + \partial_x(uq + r) = -3q\,\partial_x u + 3T\,\partial_x p,\\
&\partial_t r + \partial_x\!\Big(ur + 10\,\tfrac{pq}{\rho}\Big) = -4r\,\partial_x u + 4\,\tfrac{q}{\rho}\,\partial_x p - \partial_x \tilde X_5 .
\end{aligned}
$$

The terms inside $\partial_x(\cdot)$ on the left go through the MUSCL/Rusanov finite-volume flux (`physical_flux`). The right-hand sides and $\partial_x\tilde X_5$ are evaluated spectrally (`rhs`, `_r53_x5`). The $10pq/\rho$ term is explained in §5.4.

### 5.3 The $R_{5,3}$ closure for $X_5$ ([H19] Eq. 464)

$$
X_5 = -A_r\, v_{th}\,\mathcal H\tilde r
      \;-\; A_q\, v_{th}^2\, q
      \;-\; A_T\, n_0 v_{th}^3\,\mathcal H T
      \;-\; A_u\, n_0 T_0 v_{th}^2\, u ,
$$

with $D = 9\pi^2 - 69\pi + 128$ and

| Constant | Exact | Value |
|---|---|---|
| $A_r$ | $\dfrac{(104-33\pi)\sqrt\pi}{2D}$ | $5.13185$ |
| $A_q$ | $\dfrac{81\pi-256}{2D}$ | $-13.53748$ |
| $A_T$ | $\dfrac{3(160-51\pi)\sqrt\pi}{4D}$ | $-5.20074$ |
| $A_u$ | $\dfrac{135\pi^2-750\pi+1024}{2D}$ | $1.78706$ |

$D \approx 0.0565$ is small, so these must be computed in double precision from the exact expressions. In the code they are `_R53_Ar`, `_R53_Aq`, `_R53_AT` and `_R53_Au`.

### 5.4 Change made: splitting off the Grad part of $X_5$

The closure in §5.3 is linear, with coefficients fixed at $T_0$. I instead write

$$
X_5 = 10\,\frac{p\,q}{\rho} + \tilde X_5 ,
$$

compute the first term from local values inside the flux, and apply the linear closure only to $\tilde X_5$.

**Where the 10 comes from.** Take a Maxwellian plus the lowest-order Hermite correction that carries heat flux (Grad). With $\xi = (v-u)/\sqrt T$ and $\mathrm{He}_3(\xi) = \xi^3 - 3\xi$:

$$
f = \frac{\rho}{\sqrt{2\pi T}}\,e^{-\xi^2/2}\big[1 + a\,\mathrm{He}_3(\xi)\big].
$$

Using $\langle \xi^3\mathrm{He}_3\rangle_M = 15 - 9 = 6$ and $\langle \xi^5\mathrm{He}_3\rangle_M = 105 - 45 = 60$:

$$
q = 6a\,\rho\,T^{3/2},\qquad X_5 = 60a\,\rho\,T^{5/2}
\quad\Longrightarrow\quad X_5 = 10\,T\,q = 10\,\frac{pq}{\rho}.
$$

The same expansion gives $r = 3p^2/\rho$, because $\mathrm{He}_3$ is odd. The split is therefore the 5th-moment version of the 4-moment code's $r = 3p^2/\rho + \tilde r$.

**Keeping the linear response unchanged.** Linearised, $10pq/\rho \to 10T_0\,q^{(1)} = 5v_{th}^2 q^{(1)}$. Subtracting this from the $q$ coefficient of §5.3 gives

$$
\boxed{\;
\tilde X_5 = -A_r\, v_{th}\,\mathcal H\tilde r
             \;-\; \big(A_q v_{th}^2 + 10T_0\big)\, q
             \;-\; A_T\, n_0 v_{th}^3\,\mathcal H T
             \;-\; A_u\, n_0 T_0 v_{th}^2\, u \;}
$$

In Fourier space, as implemented:

$$
\hat{\tilde X}_{5,k} = -A_r v_{th}\, i\,\mathrm{sgn}(k)\,\hat{\tilde r}_k
 - \big(A_q v_{th}^2 + 10T_0\big)\hat q_k
 - A_T n_0 v_{th}^3\, i\,\mathrm{sgn}(k)\,\hat T_k
 - A_u n_0 T_0 v_{th}^2\,\hat u_k ,\qquad \hat{\tilde X}_{5,0} = 0 .
$$

In the code's units, $A_q v_{th}^2 + 10T_0 = (2A_q + 10)\,T_0 \approx -17.07\,T_0$.

**Why the split was needed.** The exact terms ($5r\,\partial_x u$, $4(q/\rho)\partial_x p$, and so on) scale with the local $T$, while the closure's coefficients are frozen at $T_0$. In two-stream saturation the beams heat to $T \approx 6$–$9\,T_0$. The local system then loses hyperbolicity (its characteristic speeds become complex) and the run blew up. With the split, the dominant local part of $X_5$ follows the local $T$, and only a smaller remainder is frozen at $T_0$. See §7 for the numbers.

### 5.5 Linearisation check

Linearising the $r$ equation about $u_0 = q_0 = 0$ gives $\partial_t r^{(1)} = -5r_0\,\partial_x u - \partial_x X_5^{(1)}$, with $r_0 = 3p_0^2/n_0$. The factor is $5r_0$: $r_0$ from $\partial_x(ur)$ plus $4r_0$ from the right-hand side. Combined with the $\rho$ and $p$ equations, this gives H19 Eq. 468:

$$
\partial_t \tilde r^{(1)} + \partial_x X_5^{(1)} - 3v_{th}^2\,\partial_x q^{(1)} = 0 .
$$

I checked this equation directly against the exact kinetic moments (H19 Eqs. 79–84 and 463); the residual is about $10^{-16}$.

The linearised 5-moment code reproduces H19's dynamic $R_{5,3}$ closure (Eq. 394) to about $10^{-7}$, the difference being due to the rounded constants in the paper. The comparison with the exact $R(\zeta) = 1 + \zeta Z(\zeta)$, at the same values of $n_0$ and $T_0$ as the two-stream beams:

| $\zeta$ | 5-moment code | exact $R$ | relative error |
|---|---|---|---|
| 0.1 | 0.98013 + 0.17548i | 0.98013 + 0.17548i | 0.00% |
| 0.5 | 0.57553 + 0.69011i | 0.57556 + 0.69019i | 0.01% |
| 0.9 | 0.02776 + 0.70980i | 0.02670 + 0.70964i | 0.15% |
| 1.5 | −0.28871 + 0.28025i | −0.28475 + 0.28022i | 0.99% |
| 3.0 | −0.06987 + 0.00146i | −0.06963 + 0.00066i | 1.21% |
| 0.7 − 0.4i | 0.14787 + 1.67586i | 0.15572 + 1.67723i | 0.47% |

---

## 6. Wave speeds

Rusanov flux: $F_{i+1/2} = \tfrac12(F_L + F_R) - \tfrac12\alpha\,(U_R - U_L)$. Both $\alpha$ and the CFL time step use a bound on the local characteristic speeds.

**4-moment model.** At $q = 0$ the local (non-Hilbert) system has speeds $u \pm \xi\,v_t$, where $\xi$ are the roots of $\mathrm{He}_4$, $\xi^4 - 6\xi^2 + 3 = 0$:

$$
\xi = \pm\sqrt{3\pm\sqrt6} = \pm 0.742,\ \pm 2.334 .
$$

The code uses `_4m_lam0_coef` $= 3.82$, which overestimates the fastest speed by a factor of 1.64. Tests show this does **not** change the damping rates (to the 5th digit); it only takes about 1.6× more time steps than necessary.

**5-moment model.** At $q = 0$ and $T = T_0$, the local linear system $\partial_t W + M\,\partial_x W = 0$ with $W = (\rho, u, p, q, r)$ has, in units of $v_t$,

$$
M = \begin{pmatrix}
0&1&0&0&0\\ 0&0&1&0&0\\ 0&3&0&1&0\\ 0&0&-3&0&1\\ 0&\;15-2A_u\;&0&\;-2A_q\;&0
\end{pmatrix},
\qquad \lambda = 0,\ \pm1.699,\ \pm4.918 .
$$

The code computes `_5m_coef` $= \max\lvert\lambda\rvert$ from this matrix in `__init__` and uses $\lvert u\rvert + 4.918\sqrt{T}$. In two-stream saturation the true speed exceeds this bound by up to 14% (§7).

---

## 7. Two-stream experiments

Setup: two beams at $\pm1$, $n_0 = 0.5$ each, $T_0 = 0.16$, $k = 0.5$, `nx = 128`, RK4, CFL 0.1. "Max Im" is the largest imaginary part of the local characteristic speeds over the grid.

**Original closure (no split):**

| t | $T/T_0$ range | max Im | true speed / bound |
|---|---|---|---|
| 15 | 0.69–1.56 | 0.000 | 1.13 |
| 20 | 0.38–6.35 | 0.638 | 1.29 |
| 21.5 | −3.06–8.90 | 1.206 | 1.80 |

It goes to NaN at $t \approx 22$. Making $\alpha$ and the CFL speed **2× or 4× larger** still gives NaN at $t \approx 23.6$. The wave-speed bound is **not** the cause: upwind dissipation cannot fix a system whose characteristic speeds have become complex.

**With the split (current code):**

| t | $T/T_0$ range | max Im | true speed / bound |
|---|---|---|---|
| 15 | 0.69–1.55 | 0.000 | 1.09 |
| 20 | 0.53–5.62 | 0.660 | 1.14 |
| 25 | 2.19–6.78 | 0.675 | 0.81 |
| 30 | 4.06–4.72 | 0.132 | 0.65 |

It runs to $t = 40$ with no NaN, and the linear growth rate is unchanged (0.232). **This is not a complete fix**: hyperbolicity is still lost for a while during saturation, with imaginary parts about half as large as before. The split makes the problem milder and short-lived; it does not guarantee stability in other cases.

The moments also become unrealizable during saturation: $r\rho < p^2 + \rho q^2/p$, which no non-negative distribution can satisfy.

---

## 8. Validation (nonlinear code, damping rates from the decay of $\lvert\hat E_k\rvert$)

| Case | exact kinetic $\gamma$ | 4-moment | 5-moment |
|---|---|---|---|
| Electron Landau damping, $k = 0.5$ | −0.15336 | −0.1152 (24.9%) | −0.15369 (0.2%) |
| Ion acoustic, $\tau = 0.5$, $k = 2\pi/100$ | −0.02526 | −0.01695 (32.9%) | −0.02427 (3.9%) |

The 4-moment error comes from the $R_{4,3}$ closure itself, not from the numerics. Solving the dispersion relation analytically with $R \to R_{4,3}$ gives $\gamma = -0.1127$ for the electron case, close to the code's $-0.1152$. Neither the wave-speed coefficient nor doubling `nx` changes the result.

The ion reference is the exact root of $R(\zeta) = -\tau$ (Boltzmann electrons, quasi-neutral), with $\zeta = \omega/(k\sqrt{2\tau})$.

---

## 9. Open issues and recommended next steps

1. **Realizable reconstruction.** MUSCL currently reconstructs the conserved variables, which can produce faces with $p < 0$ or $r\rho < p^2 + \rho q^2/p$. Reconstruct primitive variables $(\rho, u, T, q, \tilde r)$ instead, and fall back to first order at any face whose reconstructed state is unrealizable.
2. **Non-conservative terms.** Terms such as $-3q\,\partial_x u$ and $4(q/\rho)\,\partial_x p$ use spectral derivatives with no dissipation, so they produce Gibbs oscillations at the steep gradients of saturation. Either use limited finite-volume differences, or add a spectral filter (2/3 dealiasing or an exponential filter).
3. **Wave speeds.** For the 5-moment model, use the per-face eigenvalues of the local Jacobian (vectorised over cells) or a safety factor of about 1.2. For the 4-moment model, use $\sqrt{3+\sqrt6}$ (plus a correction for $q$) to save time steps.
4. **Galilean invariance.** The $A_u$ term in $R_{5,3}$ depends on $u$ itself. It only makes sense for the fluctuating part of $u$, which is what the code uses, since the $k = 0$ mode is removed. If this ever causes trouble, $R_{5,4}$ (H19 Eq. 465) has no $u$ term, at the cost of some accuracy ($o(\zeta^3)$, maximum error about 3.4%).
5. **NuFI comparison.** The saved 4-moment two-stream results predate the $n_0$ fix. Re-run the two-stream comparison against NuFI on the GPU machine.
