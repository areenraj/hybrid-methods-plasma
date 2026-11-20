#!/usr/bin/env python3
"""
Run the NuFI (kinetic, GPU) reference cases ONCE and cache their statistics.csv.
NuFI does not depend on the fluid model, so one NuFI run serves all 3-7 moment comparisons.

    python3 run_nufi.py [electron] [ion] [twostream] [--force]

Cached to  $MOMENTS_CACHE/nufi/<case>/statistics__<param>.csv  (default bin/moments_cache, git-ignored).
nufi/config.hpp is patched per run (as the existing sweep_*.sh scripts do) and restored at the end.
"""

import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

GEN    = os.path.join(C.HERE, "..", "generate")
BIN    = os.path.join(C.NUFI_ROOT, "bin")
CONFIG = os.path.join(C.NUFI_ROOT, "nufi", "config.hpp")


def build_and_run(gen_cmd, out, extra_args=()):
    subprocess.run([sys.executable] + gen_cmd, check=True)
    subprocess.run(["make", "-C", "nufi", f"-j{os.cpu_count()}"], cwd=C.NUFI_ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["make", "-C", "bin", "test_nufi_gpu_1d", f"-j{os.cpu_count()}"], cwd=C.NUFI_ROOT,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stats = os.path.join(BIN, "statistics.csv")
    if os.path.exists(stats):
        os.remove(stats)
    # few, passively-waiting OpenMP threads: the CPU part is small (Nx-sized Poisson + spline fit) and
    # busy-waiting threads slow NuFI down by >10x when the fluid runs share the CPU
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("NUFI_OMP_THREADS", "4"), OMP_WAIT_POLICY="PASSIVE")
    subprocess.run(["./test_nufi_gpu_1d", "--no-diagnostics", *extra_args], cwd=BIN, check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy(stats, out)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    cases = args or ["electron", "ion", "twostream"]

    jobs = []
    if "electron" in cases:
        p = C.NUFI_ELECTRON
        for k in sorted(set(C.ELECTRON_K + [C.DIAG_K])):
            jobs.append(("electron", k, [os.path.join(GEN, "gen_config_electron.py"), str(k),
                                         str(p["Nx"]), str(p["Nu"]), str(p["dt"]), str(p["T_end"])]))
    if "ion" in cases:
        p = C.NUFI_ION
        for tau in sorted(set(C.ION_TAU + [C.DIAG_TAU])):
            jobs.append(("ion", tau, [os.path.join(GEN, "gen_config_ion.py"), str(tau), str(C.ION_K),
                                      str(p["Nx"]), str(p["Nu"]), str(p["dt"]), str(p["T_end"])]))
    if "twostream" in cases:
        p = C.NUFI_TWOSTREAM
        jobs.append(("twostream", 0.0, [os.path.join(GEN, "gen_config_twostream.py"),
                                        *(str(p[key]) for key in ("k", "v0", "vth2", "alpha", "Nx", "Nu",
                                                                  "u_max", "dt", "T_end"))]))

    backup = CONFIG + ".moments_backup"
    shutil.copy(CONFIG, backup)
    try:
        for case, param, gen in jobs:
            out = C.nufi_stats_path(case, param)
            if os.path.exists(out) and not force:
                print(f"[NuFI] {case} {param}: cached")
                continue
            t0 = time.time()
            build_and_run(gen, out)
            print(f"[NuFI] {case} {param}: done in {time.time() - t0:.0f}s", flush=True)
    finally:
        shutil.copy(backup, CONFIG)   # leave config.hpp exactly as it was
        os.remove(backup)


if __name__ == "__main__":
    main()
