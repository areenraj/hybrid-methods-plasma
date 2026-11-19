# Landau-fluid moment models: code, equations, changes, and runbook

This document is the reference for the 1D electrostatic Landau-fluid solver in this repo. It is written for two readers:

- a **human** who wants the equations and the reasoning behind them (§1–§9);
- an **agent running the code on the GPU machine**, where NuFI is available (§10). Read §10 first if that is you.

Status as of 2026-11-19, branch `dev`.

**References**
- [HP90] Hammett & Perkins, PRL 64, 3019 (1990)
- [HDP92] Hammett, Dorland & Perkins, Phys. Fluids B 4, 2052 (1992)
- [H19] Hunana et al., *An introductory guide to fluid models with anisotropic temperatures, Part 2*, arXiv:1901.09360 (120 pages; PDF page = printed page). Useful pages:
  - 17–20: kinetic moments
  - 40–45: quasi-static closures and the closure table
  - 49–51: real space and the $q$ equation
  - 54–56: 5-pole closures
  - 64–70: Langmuir mode and closures for $X_5$
  - 71–72: closures for $X_6$ and $R_{7,5}$

---

## 1. Repository layout

| Path | Role |
|---|---|
| `fv_core.py` | Numerics library: `Grid` (FFT, Poisson), `FluidSolver` (all fluid models), `advance` (Euler / SSPRK3 / RK4) |
| `landau_damping.py` | Electron or ion-acoustic Landau damping driver. Writes `results/efield_data_{electron,ion}.npz` |
| `two_stream.py` | Two counter-streaming electron beams sharing one $E$ field. Writes `results/efield_data_twostream.npz` |
| `electron_damping_analysis.py`, `electron_k_sweep/`, `tau_sweep/` | Single-run analysis and local sweeps (fluid only) |
| `NumericalFlowIteration/validation/` | Scripts that run NuFI and the fluid models side by side, plus saved results. Tracked in git |
| `NumericalFlowIteration/` (everything else) | The NuFI source. **Not tracked: never commit it** |
| `Literature/` | Papers. **Not tracked: never commit it** |

**Drivers write to `results/` relative to the current working directory, and that directory must already exist** (`mkdir -p results`). The `validation/scripts/run/run_hp_*.py` wrappers create it themselves.

---

## 2. Code structure (`fv_core.py`)

### 2.1 Models and state vectors

Set `MODEL` in a driver, or pass `model=` to `FluidSolver`.

| `MODEL` string | State `U` (rows) | Closure | Section |
|---|---|---|---|
| `pressureless`, `isothermal`, `energy` | $(\rho,\rho u)$ or $(\rho,\rho u,\mathcal E)$ | none (classical) | – |
| `hammett_perkins` (3HP) | $(\rho,\rho u,p)$ | $R_{3,2}$ heat flux | §4 |
| `4moment_hammett_perkins` (4HP) | $(\rho,\rho u,p,q)$ | $R_{4,3}$ for $\tilde r$ | §5 |
| `5moment_hammett_perkins` | $(\rho,\rho u,p,q,r)$ | $R_{5,3}$ for $X_5$ | §6 |
| `6moment_hammett_perkins` | $(\rho,\rho u,p,q,r,X_5)$ | $R_{6,4}$ for $X_6$ | §6 |
| `7moment_hammett_perkins` | $(\rho,\rho u,p,q,r,X_5,X_6)$ | $R_{7,5}$ for $X_7$ | §6 |

The 5-, 6- and 7-moment models share one code path, selected through the dict `_HI_MODELS = {model: N}`. All moments are **central** moments about $u$.

### 2.2 How one RHS evaluation works (`FluidSolver.rhs`)

1. **Flux part.** `_muscl_states` does MUSCL reconstruction of the conserved `U` with a van Leer limiter. `numerical_flux` applies a Rusanov (local Lax–Friedrichs) flux built from `physical_flux` and a wave-speed bound $\alpha$. The result is $-\Delta F/\Delta x$.
2. **Field force.** Adds `charge_sign * rho * E` to the momentum row. $E$ is computed outside the solver (Poisson) and passed in.
3. **Non-conservative source terms**, such as $-l\,W_l\,\partial_x u$ and $l\,(W_{l-1}/\rho)\,\partial_x p$. Derivatives are spectral (`Grid.spectral_deriv`).
4. **Closure.** Computed spectrally: the Hilbert transform is `1j*k/|k|`, the $k=0$ mode is removed, and the derivative of the closed term is subtracted from the last row.

| Function | Purpose |
|---|---|
| `_hp_heatflux` | 3HP closure for $q$ |
| `_r4hp_r` | 4HP closure for $\tilde r$ |
| `_hi_closure(N, …)` | Linear closure for the remainder of $W_N$, for $N = 5, 6, 7$ |
| `_grad_part(N, rho, p, W)` | Local (Grad) part of $W_N$, used inside `physical_flux` |
| `_hi_speed_coef(N)` | Wave-speed bound in units of $\sqrt T$, computed once in `__init__` |
| `compute_dt` | CFL time step: `CFL * dx / max_speed` |

The closure constants are module-level: `_HP_D1`, `_HP_beta1`, `_R53_*`, `_R64_*`, `_R75_*`.

### 2.3 Normalisation and the meaning of `vt`

The code sets $m = 1$, $|q_s| = 1$ and $\omega_{p} = 1$ (for density 1). `FluidSolver(vt=...)` is the background thermal speed $v_t=\sqrt{T_0}$, so $T_0 = $ `vt**2`. **`vt` must match the initial temperature $p/\rho$ of that species**, because the closures are linearised about $T_0$:

- electron Landau damping: $p = 1$, $\rho = 1$, `vt = 1`
- ion acoustic: $p = \tau$, `vt = sqrt(tau)`
- two-stream: $p = 0.08$, $\rho = 0.5$ per beam, so $T = 0.16$ and `vt = 0.4`

$n_0$ in the closures is `np.mean(rho)` of that species.

---

## 3. Notation and the moment hierarchy

| Symbol | Code | Meaning |
|---|---|---|
| $v_t=\sqrt{T_0}$ | `self.vt` | HP90 thermal speed |
| $v_{th}=\sqrt{2T_0}=\sqrt2\,v_t$ | `np.sqrt(2)*self.vt` | H19 thermal speed (all H19 formulas use this) |
| $\mathcal H\leftrightarrow i\,\mathrm{sgn}(k)$ | `1j*k/|k|` | Hilbert transform, H19 sign convention |
| $\zeta=\omega/(\lvert k\rvert v_{th})$ | | normalised phase velocity |
| $W_l = m\int(v-u)^l f\,dv$ | `U[l]` for $l\ge2$ | central moments: $W_2=p$, $W_3=q$, $W_4=r$, $W_5=X_5$, $W_6=X_6$ |
| $T=p/\rho$, $\ \tilde r=r-3p^2/\rho$ | | |

**General hierarchy** ([HDP92] Eq. 11), exact for 1D Vlasov:

$$
\partial_t W_l + \partial_x\big(u W_l + W_{l+1}\big) = -\,l\,W_l\,\partial_x u + l\,\frac{W_{l-1}}{\rho}\,\partial_x p ,\qquad l\ge2 .
$$

The electric field enters only the momentum equation $\partial_t(\rho u)+\partial_x(\rho u^2+p)=q_s\rho E$. An $N$-moment model evolves $W_0,\dots,W_{N-1}$ and needs a closure for $W_N$.

**Exact kinetic linear response.** For a Maxwellian background, with $R(\zeta) = 1+\zeta Z(\zeta)$, the density response is $n_1/n_0=-(q\Phi/T_0)\,R(\zeta)$. Every other moment response is also a function of $\zeta$ and $R$ (H19 Eqs. 78–84 and 463).

---

## 4. How a Padé approximant $R_{n,n'}$ becomes a closure

A fluid model with $N$ equations, linearised, always gives a **rational** response $P(\zeta)/Q(\zeta)$ with $N$ poles. The exact $R(\zeta)$ is transcendental. The Padé approximant $R_{n,n'}$ is the $n$-pole rational function that matches $n'$ terms of the large-$\zeta$ series of $R$ and as many small-$\zeta$ terms as remain. More small-$\zeta$ terms means more accuracy for strongly damped modes; more large-$\zeta$ terms means a better high-phase-velocity limit.

To build a closure, substitute $R_{n,n'}$ for $R$ in the kinetic moments. Every moment becomes (polynomial)/$D$ with a common denominator. If some linear combination of moments is free of $\zeta$, that combination is the closure. By construction, the fluid model then reproduces exactly $R_{n,n'}$. This only works when roughly $n'\ge n-2$.

Example: the 3HP heat flux. With $R_{3,2}$, $T_1\propto -i\sqrt\pi\zeta/D$ and $q_1\propto -2\zeta/D$, so

$$
\hat q_k=-\tfrac{2}{\sqrt\pi}\,n_0 v_{th}\,i\,\mathrm{sgn}(k)\,\hat T_k .
$$

---

## 5. 3- and 4-moment closures

**3HP, $R_{3,2}$** ([HP90] Eq. 7, [H19] Eq. 276):

$$
\hat q_k = -2\sqrt{\tfrac{2}{\pi}}\; n_0 v_t\, i\,\mathrm{sgn}(k)\,\hat T_k .
$$

**4HP, $R_{4,3}$** ([HP90], [H19] Eqs. 292 and 324). The $q$ flux is $uq+3pT$, so $r = 3p^2/\rho+\tilde r$ with the $3p^2/\rho$ part handled locally:

$$
\hat{\tilde r}_k = -\frac{2\sqrt\pi}{3\pi-8}\, v_{th}\, i\,\mathrm{sgn}(k)\,\hat q_k + \frac{32-9\pi}{2(3\pi-8)}\, n_0 v_{th}^2\, \hat T_k
= -D_1\sqrt2\, v_t\,\mathcal H\hat q + 2\beta_1 n_0 v_t^2\hat T ,
$$

with $D_1=\frac{2\sqrt\pi}{3\pi-8}$ and $\beta_1=\frac{32-9\pi}{6\pi-16}$.

**Bug fixed (commit be995a1).** The $D_1$ term had an extra `n0` factor. It was harmless when $n_0=1$ but halved the damping of $q$ in the two-stream runs ($n_0=0.5$ per beam). Any 4HP two-stream results produced before this fix are wrong, including the saved ones in `validation/results/4moment_hammett_perkins/two_stream/`.

---

## 6. 5-, 6- and 7-moment models

### 6.1 Splitting each closed moment into a local part and a closure part

Applying H19's linear closures as written, with coefficients fixed at $T_0$, made the two-stream runs blow up during saturation (§8.3). The fix: split the closed moment as

$$
W_N = W_N^{\rm Grad}(\text{local } \rho,p,q,\dots) + \tilde W_N ,
$$

compute $W_N^{\rm Grad}$ inside the flux, and apply the linear closure only to $\tilde W_N$. The closure coefficients are shifted by the linearisation of $W_N^{\rm Grad}$ so that the **linear response is unchanged**. This is the same idea as the 4HP split $r=3p^2/\rho+\tilde r$.

$W_N^{\rm Grad}$ is the value of $W_N$ for $f = M(\xi)\,[1+a_3\mathrm{He}_3+a_4\mathrm{He}_4+a_5\mathrm{He}_5]$, with $\xi = (v-u)/\sqrt T$, where the $a_i$ are fixed by the evolved moments:

$$
X_5^{\rm Grad}=10\,T\,q,\qquad
X_6^{\rm Grad}=15\,T\,r-30\,\frac{p^3}{\rho^2}=15\frac{p^3}{\rho^2}+15\,T\,\tilde r,\qquad
X_7^{\rm Grad}=21\,T\,X_5-105\,T^2 q .
$$

### 6.2 5-moment model, $R_{5,3}$ ([H19] Eq. 464)

This is the most precise reliable closure at this level: $o(\zeta^4)$ for small $\zeta$, $o(\zeta^{-5})$ for large $\zeta$. With $D=9\pi^2-69\pi+128$:

$$
X_5=-A_r v_{th}\mathcal H\tilde r - A_q v_{th}^2 q - A_T n_0 v_{th}^3\mathcal H T - A_u n_0T_0 v_{th}^2 u,
$$

| | exact | value |
|---|---|---|
| $A_r$ | $(104-33\pi)\sqrt\pi/(2D)$ | 5.13185 |
| $A_q$ | $(81\pi-256)/(2D)$ | −13.53748 |
| $A_T$ | $3(160-51\pi)\sqrt\pi/(4D)$ | −5.20074 |
| $A_u$ | $(135\pi^2-750\pi+1024)/(2D)$ | 1.78706 |

**As implemented** (closure applied to $\tilde X_5 = X_5-10Tq$):

$$
\tilde X_5=-A_r v_{th}\mathcal H\tilde r-\big(A_q v_{th}^2+10T_0\big)q-A_T n_0 v_{th}^3\mathcal H T-A_u n_0 T_0 v_{th}^2 u .
$$

### 6.3 6-moment model, $R_{6,4}$ ([H19] Eq. 475)

H19 already writes this for $\tilde X_6=X_6-15p^3/\rho^2$. With $D_6=801\pi^2-5124\pi+8192$:

$$
\tilde X_6=-e_5 v_{th}\mathcal H X_5+e_r v_{th}^2\tilde r-e_q v_{th}^3\mathcal H q-e_T n_0 v_{th}^4 T-e_u n_0T_0 v_{th}^3\mathcal H u,
$$

| | exact |
|---|---|
| $e_5$ | $3(180\pi^2-1197\pi+1984)\sqrt\pi/D_6$ |
| $e_r$ | $3(675\pi^2-4728\pi+8192)/(2D_6)$ |
| $e_q$ | $3(285\pi-896)\sqrt\pi/(2D_6)$ |
| $e_T$ | $3(7065\pi^2-43056\pi+65536)/(4D_6)$ |
| $e_u$ | $9(450\pi^2-2799\pi+4352)\sqrt\pi/D_6$ |

**As implemented.** The local part is $15p^3/\rho^2+15T\tilde r$, so the $\tilde r$ coefficient becomes $e_r v_{th}^2-15T_0$.

### 6.4 7-moment model, $R_{7,5}$ (derived here from [H19] Eq. 478)

H19 gives $R_{7,5}$ only as a *dynamic* closure, i.e. an evolution equation for $\tilde X_6$. With $D_7=10800\pi^3-120915\pi^2+440160\pi-524288$:

$$
\partial_t\tilde X_6=-a_6 v_{th}\partial_x\mathcal H\tilde X_6+b_5v_{th}^2\partial_xX_5+b_rv_{th}^3\partial_x\mathcal H\tilde r+b_qv_{th}^4\partial_x q-b_Tn_0v_{th}^5\partial_x\mathcal HT+b_un_0T_0v_{th}^4\partial_xu ,
$$

| | exact |
|---|---|
| $a_6$ | $18(1545\pi^2-9743\pi+15360)\sqrt\pi/D_7$ |
| $b_5$ | $3(52425\pi^2-331584\pi+524288)/(2D_7)$ |
| $b_r$ | $3(7875\pi^2-50490\pi+80896)\sqrt\pi/D_7$ |
| $b_q$ | $3(162000\pi^3-1758825\pi^2+6263040\pi-7340032)/(4D_7)$ |
| $b_T$ | $27(15825\pi^2-99260\pi+155648)\sqrt\pi/(2D_7)$ |
| $b_u$ | $3(189000\pi^3-1612215\pi^2+4534656\pi-4194304)/(2D_7)$ |

(The sign convention for $\partial_x\mathcal H$ follows H19, where $\lvert k\rvert\to-\partial_x\mathcal H$.)

The exact linear equation for $\tilde X_6$ is $\partial_t\tilde X_6+\partial_xX_7-45T_0^2\partial_x q=0$. Equating the two and integrating once in $x$ gives the static closure

$$
X_7=a_6v_{th}\mathcal H\tilde X_6-b_5v_{th}^2X_5-b_rv_{th}^3\mathcal H\tilde r-\big(b_qv_{th}^4-45T_0^2\big)q+b_Tn_0v_{th}^5\mathcal HT-b_un_0T_0v_{th}^4u .
$$

**As implemented**, with local part $21TX_5-105T^2q$: the $X_5$ coefficient becomes $-(b_5v_{th}^2+21T_0)$ and the $q$ coefficient becomes $-(b_qv_{th}^4-150T_0^2)$.

### 6.5 Initial conditions

Maxwellian central moments in both drivers: $q=X_5=0$, $r=3p^2/\rho$, $X_6=15p^3/\rho^2$.

---

## 7. Wave speeds

Rusanov flux: $F=\tfrac12(F_L+F_R)-\tfrac12\alpha(U_R-U_L)$. The bound $\alpha=\lvert u\rvert+c_N\sqrt T$, where $c_N$ is the largest $\lvert\lambda\rvert$ of the local (non-Hilbert) linearised system about a Maxwellian, computed by `_hi_speed_coef`:

| Model | $c_N$ (units of $\sqrt T$) | note |
|---|---|---|
| 4HP | code uses 3.82 | exact is $\sqrt{3+\sqrt6}=2.334$ (roots of $\mathrm{He}_4$); the overestimate costs about 1.6× more steps, accuracy unaffected |
| 5-moment | 4.918 | |
| 6-moment | 6.548 | |
| 7-moment | 8.334 | each extra moment shrinks the time step by about 25% |

The bound assumes $q=0$ and $T=T_0$. In two-stream saturation the true local speed exceeded it by up to 14% for the 5-moment model.

---

## 8. Verification and results

### 8.1 Linear verification (done)

- The linearised 4HP code reproduces $R_{4,3}(\zeta)$ to machine precision.
- An independent implementation of H19 Eq. 478 reproduces H19's $R_{7,5}$ error table (0.134% / 0.156% vs 0.13 / 0.16%).
- **Eigenvalues of the actual `FluidSolver.rhs`** (finite-difference Jacobian about a uniform Maxwellian, central flux, Langmuir $k=0.5$) converge to the closures' analytic $\omega$ under grid refinement. At `nx = 1024` the difference is about $6\times10^{-6}$ for 5, 6 and 7 moments, and it shrinks with `nx`.
- The linear $r$ equation was checked against the exact kinetic moments (H19 Eq. 468); the residual is about $10^{-16}$.

### 8.2 Damping rates (nonlinear code, $\gamma$ fitted from the decay of $\lvert\hat E_k\rvert$ after its peak)

Electron: `nx = 128`, `t_end = 40`, $k=0.5$. Ion: $\tau=0.5$, $k=2\pi/100$, `L = 100`, `nx = 512`, `t_end = 450`. Errors are relative to the exact kinetic root.

| Model | Electron: closure alone | Electron: simulation | Ion: closure alone | Ion: simulation |
|---|---|---|---|---|
| exact kinetic $\gamma$ | −0.15336 | | −0.025263 | |
| 4HP | −0.11272 (26.5%) | −0.11522 (24.9%) | −0.016330 (35.4%) | −0.016950 (32.9%) |
| 5-moment | −0.14771 (3.7%) | −0.15369 (0.2%) | −0.023461 (7.1%) | −0.024267 (3.9%) |
| 6-moment | −0.14930 (2.6%) | −0.15553 (1.4%) | −0.024999 (1.0%) | −0.025708 (1.8%) |
| 7-moment | −0.15222 (0.7%) | −0.15837 (3.3%) | −0.025351 (0.3%) | −0.026048 (3.1%) |

**Read this table carefully.** The scheme adds its own damping, about 2–4% extra and roughly the same for every model (from MUSCL/Rusanov numerical diffusion). So:
- the 5-moment model's 0.2% is a lucky cancellation of a 3.7% closure error;
- from 6 moments up, the numerical damping is larger than the closure error, which is why 7 moments looks *worse* in simulation than 6.

Higher-moment gains will only show after the numerical diffusion is reduced (finer grid, primitive-variable reconstruction, or a less dissipative flux).

### 8.3 Two-stream ($\pm1$ beams, $n_0=0.5$, $T_0=0.16$ each, $k=0.5$, `nx = 128`, RK4, CFL 0.1, `t_end = 40`)

| Model | growth rate, $5<t<15$ | NaN | steps | wall time |
|---|---|---|---|---|
| 4HP | 0.2370 | none | 15 338 | 15 s |
| 5-moment | 0.2320 | none | 17 629 | 18 s |
| 6-moment | 0.2322 | none | 20 986 | 24 s |
| 7-moment | 0.2322 | **t ≈ 21.4** | 24 338 | 29 s |

References: the exact kinetic growth rate for this setup is **0.2385** (two drifting Maxwellians, root of $1+\sum_b \frac{n_b}{k^2T_b}R(\zeta_b)=0$); the cold-fluid rate is 0.3406.

**Why the runs blew up.** The closure coefficients are frozen at $T_0$, but the beams heat to 6–9 $T_0$ in saturation, and the local system loses hyperbolicity (complex characteristic speeds). Diagnosis for the 5-moment model:

| | t = 15 | t = 20 | t = 21.5 / 25 | t = 30 |
|---|---|---|---|---|
| without split: max Im $\lambda$ | 0 | 0.64 | 1.21 (NaN at ≈ 22) | – |
| with split: max Im $\lambda$ | 0 | 0.66 | 0.68 | 0.13 |

Making the Rusanov $\alpha$ 2× or 4× larger does **not** prevent the NaN, so the wave-speed bound is not the cause. The split makes the loss of hyperbolicity milder and temporary; it does not remove it. The moments also become unrealizable in saturation ($r\rho<p^2+\rho q^2/p$). **The 7-moment NaN has not yet been diagnosed**; the working hypothesis is the same mechanism with more frozen coefficients.

### 8.4 Comparison of all H19 1D closures (linear)

These are max relative errors of $R$ over real $\zeta\in(0,4)$, and $\gamma$ errors against the exact roots. The values match H19's p. 39 table to within about 10%.

| Closure | max err Im / Re | Langmuir k = 0.3 / 0.4 / 0.5 | Ion τ = 0.1 / 0.5 / 1.0 |
|---|---|---|---|
| $R_{3,2}$ (3HP) | 52 / 51% | 136 / 12 / 39% | 84 / 55 / 62% |
| $R_{4,2}$ | 5.2 / 5.6% | 162 / 30 / 3.5% | 117 / 12 / 19% |
| $R_{4,3}$ (4HP) | 11 / 13% | 81 / 10 / 27% | 49 / 35 / 40% |
| $R_{5,3}$ | 1.6 / 1.8% | 53 / 5.7 / 3.7% | 38 / 7.1 / 4.9% |
| $R_{5,4}$ | 3.3 / 3.6% | 31 / 11 / 17% | 16 / 16 / 14% |
| $R_{5,5}$ | 11 / 9% | 16 / 36 / 36% | 24 / 33 / 2.2% |
| $R_{6,4}$ | 0.47 / 0.54% | 17 / 0.6 / 2.6% | 11 / 1.0 / 2.5% |
| $R_{7,5}$ | 0.13 / 0.16% | – | – |

Notes:
- At the 4-moment level, $R_{4,2}$ is more accurate than the $R_{4,3}$ that 4HP uses. But $R_{4,2}$ fails for weakly damped modes, because its heat flux has the wrong large-$\zeta$ asymptote.
- Weakly damped modes (large $\zeta$, e.g. Langmuir $k\lambda_D\lesssim0.3$) are poorly captured by every closure: the exponentially small $e^{-\zeta^2}$ damping has no Padé representation. H19 p. 67 recommends Langmuir runs only for $k\lambda_D>0.2$.

---

## 9. Known issues and next steps

1. **7-moment two-stream NaN at t ≈ 21.4.** Diagnose it the way §8.3 did: track local eigenvalues and realizability through saturation.
2. **Make the closure coefficients follow the local temperature** ($v_{th}\to\sqrt{2T}$) instead of $T_0$. This addresses the root cause of hyperbolicity loss for all models.
3. **Reduce numerical diffusion.** Reconstruct primitive variables $(\rho,u,T,q,\tilde r,\dots)$ with a realizability fallback, or filter the spectral derivatives in the non-conservative terms (they are undamped and ring at steep gradients).
4. **4HP speed coefficient:** use $\sqrt{3+\sqrt6}$ to save time steps.
5. **$A_u$ / $e_u$ / $b_u$ terms depend on $u$ itself.** They are only meaningful for fluctuations; the $k=0$ mode is removed, so the mean flow is not seen.
6. **Two-stream initial condition differs from NuFI's** (§10.4). Fix that before comparing.
7. **Nonlinear trapping physics** (O'Neil oscillations, phase-space holes) is outside what any of these closures can represent. Expect agreement with NuFI in the linear phase and the saturation level, not in post-saturation dynamics.

---

## 10. Runbook for the agent on the GPU machine

### 10.1 What you are expected to do

1. Pull branch `dev`. The fluid code is pure Python: `numpy`, `scipy`, `matplotlib`. `plasmadisp` is needed only by the old analysis scripts, and `ffmpeg` only for movies.
2. Run NuFI (kinetic reference) and the fluid models for: electron Landau damping (sweep $k$), ion-acoustic damping (sweep $\tau$), and the two-stream instability.
3. Produce comparison plots and tables, and report the numbers back (§10.6).

**Do not commit NuFI source, NuFI binaries or build output, `Literature/`, or the large NuFI dumps (`f_*.txt`, `phase_flow_*.txt`, `coeffs_*.txt`).** Stage files explicitly; never use `git add -A`. A root `.gitignore` that excludes `Literature` and `NumericalFlowIteration` exists locally on the laptop but is not committed. The validation scripts and results under `NumericalFlowIteration/validation/` *are* tracked.

### 10.2 NuFI on the GPU machine

Build (needs autotools, CUDA, BLAS/LAPACK, FFTW, MPI):

```bash
cd NumericalFlowIteration
autoreconf --install && ./configure && make
```

The validation scripts expect the GPU machine's **modified** NuFI, which differs from upstream:
- `bin/test_nufi_gpu_1d` writes `statistics.csv` with **8 columns**: `Time; L1; L2; Electric Energy; Kinetic Energy; Total Energy; Entropy; E_kmode_amp`. `E_kmode_amp` is already the physical $\lvert E_k\rvert$ amplitude of the $k$ mode.
- `config.hpp` has `ion_acoustic` and `charge_sign` fields (quasineutral Boltzmann-electron Poisson for the ion runs), and the driver accepts `--no-diagnostics`.

If your NuFI lacks these, the ion sweep and `extract_gamma_nufi.py` will not work as written. Check before running.

`validation/scripts/generate/gen_config_{electron,ion}.py` patch `nufi/config.hpp` in place, rewriting the `dim1` constructor and `f0`. **`libnufi.a` must be rebuilt after every patch** (`make -C nufi`), because `f0` is compiled into the CUDA kernel. The sweep scripts already do this.

### 10.3 Damping sweeps (fluid + NuFI)

```bash
cd NumericalFlowIteration/validation/scripts
HP_MODEL=6moment_hammett_perkins bash sweep_electron_k.sh
HP_MODEL=6moment_hammett_perkins bash sweep_ion_tau.sh
```

- Valid `HP_MODEL` values: `hammett_perkins`, `4moment_hammett_perkins`, `5moment_hammett_perkins`, `6moment_hammett_perkins`, `7moment_hammett_perkins`.
- `run_all_sweeps.sh` currently loops only over the first two. Extend its `MODELS=(...)` array to include the new ones.
- Results go to `validation/results/<model>/`.
- The fluid side is cheap. Per point, roughly: electron 2–11 s, ion 20–60 s, rising with moment count.
- The electron sweep starts at $k=0.23$, close to the $k\lambda_D>0.2$ validity limit. Expect large errors for every fluid model at the lowest $k$; that is physics, not a bug.

**Expected outcome.** For ion $\tau$ in 0.1–1.25 and electron $k$ in 0.3–0.65: 6-moment within a few % of NuFI over most of the range; 4HP 25–40% under-damped; 7-moment not better than 6 (numerical damping dominates, §8.2). If NuFI disagrees with the exact kinetic root (§8.2), suspect NuFI resolution (`Nu`, `T_END` vs recurrence time $2\pi/(k\,\Delta v)$) first.

### 10.4 Two-stream comparison: match the initial condition first

The saved NuFI two-stream run (`validation/results/4moment_hammett_perkins/two_stream/statistics.csv`) and `two_stream.py` **start differently**:

| | NuFI saved run | `two_stream.py` |
|---|---|---|
| perturbation | **density**, $\alpha=0.001$ | **velocity**, $u_{1,2}=\pm(1+0.01\cos kx)$, uniform density |
| $\lvert E_k\rvert$ at t = 0 | 0.002 | 0 |
| early $\lvert E_k\rvert$ | drops to about $10^{-5}$ at t ≈ 3.3, then grows | rises from 0 to about 0.004 by t = 1 |
| background | consistent with $\pm1$ beams, $T=0.16$ ($\langle v^2\rangle = 1.16$) | $\pm1$ beams, $T=0.16$ |

So the curves are offset in time and amplitude even though the growth rates agree. The density seed also excites the stable modes alongside the growing one, which makes the NuFI growth-rate fit sensitive to the window (0.21–0.33 depending on the window).

To compare properly:
1. Confirm NuFI's actual two-stream `f0` in `nufi/config.hpp` on the GPU machine. The laptop copy is set to a different test.
2. Make `two_stream.py` use the same seed. Set `u1 = +v_b`, `u2 = -v_b`, and give both beams `rho = 0.5*(1 + alpha*cos(k*x))` with NuFI's $\alpha$, keeping `p = 0.08`.
3. Re-run every fluid model (4HP's old saved result predates the `n0` fix, §5). Then plot $\lvert E_k(t)\rvert$ for NuFI and all fluid models on one semilog axis, using NuFI column 8 and the fluid `abs(E_kmode_real + 1j*E_kmode_imag) * 2/nx`.
4. Fit the growth rate in the clean exponential window for each run, and compare with the kinetic 0.2385.

`validation/scripts/two_stream/compare_nufi.py` and `two_stream_comparison.py` show the data-loading patterns; they hard-code the 4HP folder.

**Expected:**
- linear growth rates of 0.232–0.237 for all fluid models against 0.2385 kinetic;
- the 7-moment run goes to NaN at t ≈ 21–22;
- saturation level roughly comparable to NuFI;
- after saturation, fluid and kinetic will differ (no trapping in the fluid, §9.7).

### 10.5 Running fluid cases directly

```python
import numpy as np, landau_damping as sim          # run from repo root; mkdir -p results first
sim.MODEL = "6moment_hammett_perkins"; sim.ionDamping = False
sim.k = 0.5; sim.L = 2*np.pi/sim.k; sim.nx = 128; sim.t_end = 40.0; sim.plot_interval = 10**9
sim.main()                                         # -> results/efield_data_electron.npz
```

For ions: `sim.ionDamping = True; sim.tau = 0.5; sim.L = 100; sim.k = 2*np.pi/100; sim.nx = 512; sim.t_end = 450.0`.
For two-stream: `import two_stream as ts; ts.MODEL = ...; ts.t_end = 40.0; ts.main()`.

Fit $\gamma$ from $\lvert E_k\rvert$ after its peak, the same way as `tau_sweep/extract_gamma.py`. NuFI's `extract_gamma_nufi.py` uses the electric energy, $\gamma=\tfrac12\,d\ln E_{el}/dt$, and the two estimates agree.

### 10.6 What to report back

- Tables of $\gamma$ for NuFI, exact kinetic, and each fluid model (4HP, 5, 6, 7) over the $k$ and $\tau$ sweeps.
- The two-stream $\lvert E_k(t)\rvert$ overlay with matched initial conditions; fitted growth rates; saturation amplitude and time; NaN times.
- The NuFI resolution used (`Nx`, `Nu`, `dt`, `T_END`) and its exact `f0`.
- Anything that contradicts §8. In particular, if the 6-moment model is *not* within a few % of NuFI in the damped regime, report the parameters.
