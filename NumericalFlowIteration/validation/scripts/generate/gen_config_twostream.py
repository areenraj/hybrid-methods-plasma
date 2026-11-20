#!/usr/bin/env python3
"""
Patch nufi/config.hpp for the two-stream instability (two drifting Maxwellians).

Usage:
    python3 gen_config_twostream.py <k> <v0> <vth2> <alpha> <Nx> <Nu> <u_max> <dt> <T_end>

f0 = c [exp(-(u-v0)^2/(2 vth2)) + exp(-(u+v0)^2/(2 vth2))] (1 + alpha cos kx),
c = 1/(2 sqrt(2 pi vth2)), domain [0, 2pi/k], velocity [-u_max, u_max].
With k=0.5 v0=1 vth2=0.16 alpha=0.001 Nx=64 Nu=512 u_max=4 dt=0.1 T_end=40 this reproduces
validation/results/4moment_hammett_perkins/two_stream/statistics.csv.
"""

import math
import os
import re
import sys


def main():
    if len(sys.argv) != 10:
        print(__doc__)
        sys.exit(1)
    k, v0, vth2, alpha = (float(a) for a in sys.argv[1:5])
    Nx, Nu = int(sys.argv[5]), int(sys.argv[6])
    u_max, dt, T_end = (float(a) for a in sys.argv[7:10])
    Nt = int(round(T_end / dt))
    c = 1.0 / (2.0 * math.sqrt(2.0 * math.pi * vth2))

    config_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..", "..", "nufi", "config.hpp"))
    with open(config_path) as fh:
        src = fh.read()

    new_ctor = f"""\
template <typename real>
config_t<real>::config_t() noexcept
{{
    Nx = {Nx};
    Nu = {Nu};
    u_min = {-u_max:.17g};
    u_max =  {u_max:.17g};
    x_min = 0;
    x_max = {2.0 * math.pi / k:.17g};

    dt = {dt}; Nt = {Nt};

    Lx = x_max - x_min; Lx_inv = 1/Lx;
    dx = Lx/Nx; dx_inv = 1/dx;
    du = (u_max - u_min)/Nu;
}}"""
    src = re.sub(r"template\s*<\s*typename\s+real\s*>\s*"
                 r"config_t<real>::config_t\(\)\s*noexcept\s*\{[^}]*\}",
                 new_ctor, src, count=1)

    new_f0 = f"""\
template <typename real>
__host__ __device__
real config_t<real>::f0( real x, real u ) noexcept
{{
    using std::cos;
    using std::exp;

    // Two-stream: beams at +-v0={v0:.6g}, vth^2={vth2:.6g}, density seed alpha={alpha:.6g}, k={k:.6g}
    constexpr real v0    = {v0:.17g};
    constexpr real vth2  = {vth2:.17g};
    constexpr real alpha = {alpha:.17g};
    constexpr real k     = {k:.17g};
    constexpr real c     = {c:.17g};
    return c * ( exp( -(u-v0)*(u-v0) / (2*vth2) )
               + exp( -(u+v0)*(u+v0) / (2*vth2) ) )
             * ( 1. + alpha*cos(k*x) );
}}"""
    src = re.sub(r"template\s*<\s*typename\s+real\s*>\s*"
                 r"__host__\s+__device__\s*\n"
                 r"real\s+config_t<real>::f0\(\s*real\s+x,\s*real\s+u\s*\)\s*noexcept\s*\{.*?\}",
                 new_f0, src, count=1, flags=re.DOTALL)

    with open(config_path, "w") as fh:
        fh.write(src)


if __name__ == "__main__":
    main()
