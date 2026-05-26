#!/usr/bin/env python3
# ============================================================================
#  REPRODUCE_PAPER.PY
#
#  Single-script reproduction of:
#      "Pore-Wall Specularity Suppresses Thermal Rectification in Asymmetric
#       Nanoporous Silicon: A Nonlinear Gray Phonon-BTE Study"
#
#  WHAT THIS SCRIPT DOES
#  ---------------------
#  Stage 1 (simulations):
#      Calls bte_specularity_scan.py many times with different parameters.
#      Each call writes:
#          * one summary.csv with scalar results (Jf, Jr, R, ...)
#          * one .npz file per p_spec containing the full temperature
#            fields (Tf, Tr), the pore mask, and the convergence residuals.
#      Naming pattern of every .npz file (set inside the solver):
#          xc{xc:.3f}_r{rfrac:.3f}_a{alpha:.3f}_p{pspec:.3f}.npz
#
#  Stage 2 (aggregation):
#      Merges the per-case summary.csv files into one CSV per sweep so the
#      figure script can read them in one shot.
#
#  Stage 3 (figures):
#      Calls generate_pub_figures.py, which loads the NPZ field files and
#      the aggregated CSVs and writes all 9 publication figures to
#      ./pub_figures/ as PDF + PNG.
#
#  Stage 4 (validation analyses):
#      Runs two standalone analysis scripts that reproduce the manuscript
#      tables added during revision:
#        analysis/grid_convergence_transverse_flux.py  -> Table 5
#        analysis/gradient_orientation_analysis.py     -> Table 4
#
#  USAGE
#  -----
#  In a notebook cell:           %run reproduce_paper.py
#  From the command line:        python reproduce_paper.py
#  Quick smoke-test only:        python reproduce_paper.py --quick
#  Skip sweeps already done:     (this is the default; re-running skips
#                                 any sweep whose summary.csv exists)
#
#  EXPECTED RUNTIME
#  ----------------
#  Full reproduction:   roughly 12–24 hours on a single modern CPU core,
#                       dominated by the Nx=Ny=141, Nd=32 hi-fi runs.
#  Quick mode:          ~30–60 minutes (baseline + p_scan only).
# ============================================================================

import os
import sys
import glob
import time
import shlex
import argparse
import subprocess
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SOLVER        = ROOT / "bte_specularity_scan.py"          # rename your solver to this
FIG_SCRIPT    = ROOT / "generate_pub_figures.py"          # rename your figure script to this
RESULTS_ROOT  = ROOT / "all_results"
RESULTS_ROOT.mkdir(exist_ok=True)

# Baseline geometry used everywhere unless overridden
BASELINE = dict(xc=0.25, rfrac=0.23, alpha=0.5)

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def run_solver(outdir, Nx=101, Ny=101, Nd=16,
               xc=0.25, rfrac=0.23, alpha=0.5,
               p_spec=None, p_list=None,
               max_iter=6000, tol=2e-6,
               skip_if_done=True):
    """
    Invoke the solver once. The solver writes:
        outdir/summary.csv
        outdir/<stem>.npz   (one per p_spec value)

    `skip_if_done=True` means: if outdir/summary.csv already exists,
    skip this run. This makes the whole script idempotent — re-running
    only fills in missing sweeps.
    """
    outdir = Path(outdir)
    if skip_if_done and (outdir / "summary.csv").exists():
        print(f"[skip] {outdir.relative_to(ROOT)}  (summary.csv already present)")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(SOLVER),
           "--Nx", str(Nx), "--Ny", str(Ny), "--Nd", str(Nd),
           "--xc", str(xc), "--rfrac", str(rfrac),
           "--alpha", str(alpha),
           "--max_iter", str(max_iter), "--tol", str(tol),
           "--outdir", str(outdir)]

    if p_list is not None:
        cmd += ["--p_list", ",".join(f"{p}" for p in p_list)]
    elif p_spec is not None:
        cmd += ["--p_spec", str(p_spec)]
    else:
        raise ValueError("Must supply either p_spec or p_list")

    print(f"[run ] {outdir.relative_to(ROOT)}")
    print("       " + " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[done] {outdir.relative_to(ROOT)}  ({time.time()-t0:.1f} s)\n")


def merge_summaries(pattern, output_csv):
    """Glob per-case summary.csv files and concatenate into one CSV."""
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[warn] no summary files match: {pattern}")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[agg ] {output_csv.relative_to(ROOT)}  ({len(df)} rows)")

# ===========================================================================
# STAGE 1 — RUN ALL PARAMETER SWEEPS
# ===========================================================================
def run_all_sweeps(quick=False):
    """All simulation sweeps used by the manuscript, with one block per figure."""

    # -----------------------------------------------------------------------
    # Figure 3 — baseline |R| vs p_spec  (5 values for a smooth curve)
    # -----------------------------------------------------------------------
    run_solver(RESULTS_ROOT / "results_pscan_fine",
               p_list=[0.0, 0.25, 0.5, 0.75, 1.0],
               **BASELINE)

    # -----------------------------------------------------------------------
    # Figure 5 — temperature fields at p=0 and p=1 (the NPZ files that
    # the figure script reads for the Tf/Tr maps).
    # The NPZ files end up at:
    #     all_results/results_fields_compare/p_0.0/xc0.250_r0.230_a0.500_p0.000.npz
    #     all_results/results_fields_compare/p_1.0/xc0.250_r0.230_a0.500_p1.000.npz
    # -----------------------------------------------------------------------
    for p in (0.0, 1.0):
        run_solver(RESULTS_ROOT / "results_fields_compare" / f"p_{p}",
                   p_spec=p, **BASELINE)

    if quick:
        print("\n[quick mode] skipping xc / rfrac / alpha / convergence sweeps.\n")
        return

    # -----------------------------------------------------------------------
    # Figure 6 — |R| vs pore position
    # -----------------------------------------------------------------------
    for xc in (0.25, 0.30, 0.35, 0.40, 0.45):
        run_solver(RESULTS_ROOT / "results_xc_scan" / f"xc_{xc}",
                   xc=xc, rfrac=0.20, alpha=0.5,
                   p_list=[0.0, 0.5, 1.0])

    # -----------------------------------------------------------------------
    # Figure 7 — |R| vs porosity
    # -----------------------------------------------------------------------
    for rf in (0.10, 0.15, 0.20, 0.23):
        run_solver(RESULTS_ROOT / "results_rfrac_scan" / f"rfrac_{rf}",
                   xc=0.25, rfrac=rf, alpha=0.5,
                   p_list=[0.0, 0.5, 1.0])

    # -----------------------------------------------------------------------
    # Figure 8 — |R| vs nonlinearity strength alpha
    # -----------------------------------------------------------------------
    for a in (0.0, 0.1, 0.2, 0.3, 0.5):
        run_solver(RESULTS_ROOT / "results_alpha_scan" / f"alpha_{a}",
                   xc=0.25, rfrac=0.23, alpha=a,
                   p_list=[0.0, 0.5, 1.0],
                   max_iter=8000)

    # -----------------------------------------------------------------------
    # Figure 9 + Table 5 — grid convergence and suppression-ratio robustness
    # Needs BOTH p=0 and p=1 at four spatial resolutions, Nd=32, tight tol.
    # -----------------------------------------------------------------------
    for N in (81, 101, 121, 141):
        for p in (0.0, 1.0):
            run_solver(RESULTS_ROOT / "results_grid_pscan" / f"N_{N}" / f"p_{p}",
                       Nx=N, Ny=N, Nd=32,
                       p_spec=p,
                       max_iter=10000, tol=1e-6,
                       **BASELINE)

    # Angular convergence at fixed Nx=Ny=101, p=1
    for Nd in (16, 24, 32):
        run_solver(RESULTS_ROOT / "results_angle_scan" / f"Nd_{Nd}",
                   Nx=101, Ny=101, Nd=Nd,
                   p_spec=1.0,
                   max_iter=10000, tol=1e-6,
                   **BASELINE)

    # -----------------------------------------------------------------------
    # Sanity checks reported in Section 3.1 / Figure 2
    # (centered geometry gives |R| at machine-precision; linear and
    # off-center diffuse cases give the known reference values.)
    # -----------------------------------------------------------------------
    sanity = [
        dict(name="center_alpha0",    xc=0.50, rfrac=0.20, alpha=0.0),
        dict(name="offcenter_alpha0", xc=0.25, rfrac=0.23, alpha=0.0),
        dict(name="center_alpha05",   xc=0.50, rfrac=0.20, alpha=0.5),
    ]
    for case in sanity:
        for p in (0.0, 0.5, 1.0):
            run_solver(RESULTS_ROOT / "results_sanity_p" / f"{case['name']}_p_{p}",
                       xc=case["xc"], rfrac=case["rfrac"], alpha=case["alpha"],
                       p_spec=p, max_iter=8000)


# ===========================================================================
# STAGE 2 — AGGREGATE PER-CASE SUMMARIES INTO ONE CSV PER SWEEP
# ===========================================================================
def aggregate_all():
    """Concatenate the small per-case summary.csv files into per-sweep CSVs."""

    merge_summaries(
        pattern=str(RESULTS_ROOT / "results_xc_scan/xc_*/summary.csv"),
        output_csv=RESULTS_ROOT / "results_xc_scan/summary_all.csv")

    merge_summaries(
        pattern=str(RESULTS_ROOT / "results_rfrac_scan/rfrac_*/summary.csv"),
        output_csv=RESULTS_ROOT / "results_rfrac_scan/summary_rfrac_scan.csv")

    merge_summaries(
        pattern=str(RESULTS_ROOT / "results_alpha_scan/alpha_*/summary.csv"),
        output_csv=RESULTS_ROOT / "results_alpha_scan/summary_alpha_scan.csv")

    merge_summaries(
        pattern=str(RESULTS_ROOT / "results_grid_pscan/N_*/p_*/summary.csv"),
        output_csv=RESULTS_ROOT / "results_grid_pscan/summary_grid_pscan.csv")

    merge_summaries(
        pattern=str(RESULTS_ROOT / "results_angle_scan/Nd_*/summary.csv"),
        output_csv=RESULTS_ROOT / "results_angle_scan/summary_angle_scan.csv")

    merge_summaries(
        pattern=str(RESULTS_ROOT / "results_sanity_p/*/summary.csv"),
        output_csv=RESULTS_ROOT / "results_sanity_p/summary_sanity_p.csv")


# ===========================================================================
# STAGE 3 — RUN THE FIGURE SCRIPT
# ===========================================================================
def run_figures():
    if not FIG_SCRIPT.exists():
        print(f"[warn] {FIG_SCRIPT.name} not found; skipping figure generation.")
        return
    print(f"\n[run ] {FIG_SCRIPT.name}")
    subprocess.run([sys.executable, str(FIG_SCRIPT)], check=True, cwd=str(ROOT))
    print(f"[done] figures written to {ROOT/'pub_figures'}")


# ===========================================================================
# STAGE 4 — RUN VALIDATION ANALYSES (Tables 4 and 5 of the manuscript)
# ===========================================================================
def run_validation_analyses():
    """Reproduces the revision-response validation tables (Tables 4 and 5)."""
    analysis_dir = ROOT / "analysis"
    grid_script = analysis_dir / "grid_convergence_transverse_flux.py"
    grad_script = analysis_dir / "gradient_orientation_analysis.py"

    if grid_script.exists():
        print(f"\n[run ] {grid_script.name}  (Table 5)")
        subprocess.run([sys.executable, str(grid_script),
                        "--solver", str(SOLVER)],
                       check=True, cwd=str(ROOT))
    else:
        print(f"[warn] {grid_script} not found; skipping Table 5 analysis.")

    if grad_script.exists():
        print(f"\n[run ] {grad_script.name}  (Table 4)")
        subprocess.run([sys.executable, str(grad_script)],
                       check=True, cwd=str(ROOT))
    else:
        print(f"[warn] {grad_script} not found; skipping Table 4 analysis.")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only run baseline + p_spec scan + field comparison (~30–60 min).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the figure generation stage.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every sweep even if its summary.csv already exists.")
    args = parser.parse_args()

    # Sanity checks before starting
    if not SOLVER.exists():
        sys.exit(f"ERROR: solver not found at {SOLVER}\n"
                 f"       Rename your solver file to bte_specularity_scan.py "
                 f"or edit SOLVER at the top of this script.")

    if args.force:
        # Hack: temporarily disable the skip logic for this run
        global run_solver
        original = run_solver
        def run_solver(*a, **kw):
            kw["skip_if_done"] = False
            return original(*a, **kw)

    t0 = time.time()
    print(f"=== Stage 1: simulations  (quick={args.quick}) ===\n")
    run_all_sweeps(quick=args.quick)

    print(f"\n=== Stage 2: aggregating per-sweep summaries ===\n")
    aggregate_all()

    if not args.no_figures:
        print(f"\n=== Stage 3: generating figures ===\n")
        run_figures()

        print(f"\n=== Stage 4: validation analyses (Tables 4 and 5) ===\n")
        run_validation_analyses()

    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
