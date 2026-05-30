#!/usr/bin/env python3
"""
Patch nufi/config.hpp for ion plasma oscillations.

Usage:
    python3 gen_config_ion.py <tau> <k> <Nx> <Nu> <dt> <T_end>

tau = T_0i / T_0e  (ion-to-electron temperature ratio)
k   = wavenumber

Sets:
  - f0 = ion Maxwellian at temperature T_0i = tau (m_i=1, T_0e=1 units)
        with small cosine perturbation at wavenumber k
  - domain [0, 2pi/k]
  - velocity range scaled by sqrt(tau): [-umax*sqrt(tau), umax*sqrt(tau)]
"""

import sys
import re
import math
import os

def main():
    if len(sys.argv) != 7:
        print("Usage: gen_config_ion.py <tau> <k> <Nx> <Nu> <dt> <T_end>")
        sys.exit(1)

    tau    = float(sys.argv[1])
    k      = float(sys.argv[2])
    Nx     = int(sys.argv[3])
    Nu     = int(sys.argv[4])
    dt     = float(sys.argv[5])
    T_end  = float(sys.argv[6])

    Nt     = int(round(T_end / dt))
    x_max  = 2.0 * math.pi / k
    # ion thermal velocity = sqrt(tau); capture ±6 vth
    vth    = math.sqrt(tau)
    u_max  =  6.0 * vth
    u_min  = -6.0 * vth
    # normalisation: 1/sqrt(2*pi*tau)
    norm   = 1.0 / math.sqrt(2.0 * math.pi * tau)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "..", "nufi", "config.hpp")
    config_path = os.path.normpath(config_path)

    with open(config_path, "r") as fh:
        src = fh.read()

    # --- Replace constructor body for dim1::config_t ---
    new_ctor = f"""\
template <typename real>
config_t<real>::config_t() noexcept
{{
    Nx = {Nx};
    Nu = {Nu};
    u_min = {u_min:.17g};
    u_max =  {u_max:.17g};
    x_min = 0;
    x_max = {x_max:.17g};

    dt = {dt}; Nt = {Nt};

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;

    ion_acoustic = true;  // use quasineutral Boltzmann-electron Poisson
    charge_sign  = real(1);  // ions: forward a = -dphi/dx = E_field
}}"""

    src = re.sub(
        r"template\s*<\s*typename\s+real\s*>\s*"
        r"config_t<real>::config_t\(\)\s*noexcept\s*\{[^}]*\}",
        new_ctor,
        src,
        count=1,
    )

    # --- Replace f0 for ion Maxwellian ---
    new_f0 = f"""\
template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real u ) noexcept
{{
    using std::cos;
    using std::exp;

    // Ion Maxwellian: tau={tau:.6g}, k={k:.17g}
    constexpr real alpha = 0.01;
    constexpr real k     = {k:.17g};
    constexpr real tau   = {tau:.17g};
    constexpr real norm  = {norm:.17g};
    return norm * ( 1. + alpha*cos(k*x) ) * exp( -u*u / (2.*tau) );
}}"""

    src = re.sub(
        r"template\s*<\s*typename\s+real\s*>\s*"
        r"__host__\s+__device__\s*\n"
        r"real\s+config_t<real>::f0\(\s*real\s+x,\s*real\s+u\s*\)\s*noexcept\s*\{.*?\}",
        new_f0,
        src,
        count=1,
        flags=re.DOTALL,
    )

    with open(config_path, "w") as fh:
        fh.write(src)


if __name__ == "__main__":
    main()
