#!/usr/bin/env python3
"""
Patch nufi/config.hpp for electron Landau damping.

Usage:
    python3 gen_config_electron.py <k> <Nx> <Nu> <dt> <T_end>

Sets:
  - f0 = Maxwellian with small cosine perturbation at wavenumber k
  - domain [0, 2pi/k] (one wavelength)
  - velocity [-umax, umax] with umax = 8 (captures >99.99% of Maxwellian)
"""

import sys
import re
import math
import os

def main():
    if len(sys.argv) != 6:
        print("Usage: gen_config_electron.py <k> <Nx> <Nu> <dt> <T_end>")
        sys.exit(1)

    k      = float(sys.argv[1])
    Nx     = int(sys.argv[2])
    Nu     = int(sys.argv[3])
    dt     = float(sys.argv[4])
    T_end  = float(sys.argv[5])

    Nt     = int(round(T_end / dt))
    x_max  = 2.0 * math.pi / k
    u_max  = 8.0
    u_min  = -u_max

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "nufi", "config.hpp")
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
    u_min = {u_min};
    u_max =  {u_max};
    x_min = 0;
    x_max = {x_max:.17g};

    dt = {dt}; Nt = {Nt};

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;
}}"""

    src = re.sub(
        r"template\s*<\s*typename\s+real\s*>\s*"
        r"config_t<real>::config_t\(\)\s*noexcept\s*\{[^}]*\}",
        new_ctor,
        src,
        count=1,
    )

    # --- Replace f0 for electron Landau damping (pure Maxwellian + cosine) ---
    new_f0 = f"""\
template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real u ) noexcept
{{
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = {k:.17g};
    return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. );
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
