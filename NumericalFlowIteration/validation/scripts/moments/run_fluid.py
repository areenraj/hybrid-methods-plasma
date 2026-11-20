#!/usr/bin/env python3
"""
Run all fluid cases (3-7 moment models) in parallel, one subprocess per case.

    python3 run_fluid.py [--jobs N] [--force] [electron] [ion] [diag] [twostream]

  electron  : k sweep, all 5 models, driver IC as-is
  ion       : tau sweep, all 5 models, driver IC as-is
  diag      : startup check at electron k=0.5 and ion tau=0.5, all 5 models, IC as-is AND matched
  twostream : 4-7 moment models, IC as-is AND matched to NuFI
Traces go to $MOMENTS_CACHE/fluid/<case>/ (git-ignored).
"""

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

WORKER = os.path.join(C.HERE, "fluid_worker.py")


def job_list(groups):
    jobs = []
    if "electron" in groups:
        jobs += [("electron", m, k, "asis") for k in C.ELECTRON_K for m in C.MODELS_LANDAU]
    if "ion" in groups:
        jobs += [("ion", m, tau, "asis") for tau in C.ION_TAU for m in C.MODELS_LANDAU]
    if "diag" in groups:
        for ic in ("asis", "matched"):
            jobs += [("electron", m, C.DIAG_K, ic) for m in C.MODELS_LANDAU]
            jobs += [("ion", m, C.DIAG_TAU, ic) for m in C.MODELS_LANDAU]
    if "twostream" in groups:
        jobs += [("twostream", m, 0.0, ic) for ic in ("asis", "matched") for m in C.MODELS_TWOSTREAM]
    # de-duplicate, longest (high-moment ion) first for better load balance
    seen, out = set(), []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            out.append(j)
    out.sort(key=lambda j: ({"ion": 0, "twostream": 1, "electron": 2}[j[0]], -int(j[1][0]) if j[1][0].isdigit() else -3))
    return out


def run(job):
    case, model, param, ic = job
    out = C.fluid_trace_path(case, model, param, ic)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run([sys.executable, WORKER, case, model, str(param), ic, out],
                       env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return f"FAILED {job}: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode}"
    return p.stdout.strip()


def main():
    argv = sys.argv[1:]
    n_jobs = 12
    if "--jobs" in argv:
        i = argv.index("--jobs")
        n_jobs = int(argv[i + 1])
        del argv[i:i + 2]
    force = "--force" in argv
    groups = [a for a in argv if not a.startswith("--")] or ["electron", "ion", "diag", "twostream"]

    jobs = [j for j in job_list(groups) if force or not os.path.exists(C.fluid_trace_path(*j))]
    print(f"[fluid] {len(jobs)} runs on {n_jobs} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(n_jobs) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            print(f"[fluid {n}/{len(jobs)} {time.time() - t0:6.0f}s] {f.result()}", flush=True)


if __name__ == "__main__":
    main()
