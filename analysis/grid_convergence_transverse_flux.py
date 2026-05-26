#!/usr/bin/env python3
# ============================================================================
#  GRID CONVERGENCE + TRANSVERSE-FLUX ANALYSIS
#
#  Produces the data behind Table 5 in the manuscript and addresses:
#    - Reviewer 1 Comment 7  (Richardson-style grid convergence of the
#                             suppression ratio S)
#    - Reviewer 2 Comment 6  (spurious y-direction flux check)
#    - Reviewer 3 Comment 4  (mesh-topology robustness)
#
#  WHAT THIS SCRIPT DOES
#  ---------------------
#  For each grid resolution N = 81, 101, 121, 141 (Nd = 16 by default),
#  it solves the baseline geometry (xc=0.25, R/Lx=0.23, alpha=0.5) at
#  both p_spec = 0 and p_spec = 1, computes both qx and qy fields, then
#  builds:
#    (a) the absolute rectification magnitudes |R|_p=0 and |R|_p=1
#    (b) the relative suppression ratio S = (|R|_0 - |R|_1) / |R|_0
#    (c) the transverse-flux normalization max |<qy>| / |<qx>|
#
#  Outputs:
#    grid_p0_p1_with_transverse_flux.csv
#    summary_suppression_ratio_grid_qy.csv     <-- Table 5
#    Fig_revision_grid_suppression_qy.{png,pdf}
#
#  USAGE
#  -----
#    python analysis/grid_convergence_transverse_flux.py
#    python analysis/grid_convergence_transverse_flux.py --Nd 32   (slower, tighter)
#    python analysis/grid_convergence_transverse_flux.py --grids 81 101    (quick test)
# ============================================================================

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Build a flux-aware solver wrapper that reuses the user's BTE module.
# We need qy in addition to the standard qx (which the production solver
# already computes), so we replicate the sweep loop and add a qy reducer.
# ---------------------------------------------------------------------------
def solve_with_qy(bte, problem, TL, TR, xc, rfrac, alpha,
                  p_spec, max_iter, tol):
    """Same physics as bte.solve_2D_pore, but also returns <qy> for the
    transverse-flux diagnostic. Inputs and outputs are nondimensional."""
    (Lx, Ly, Nx, Ny, x, y, dx, dy, X, Y,
     Lambda0, T_ref, Nd, dirs, w) = problem

    is_pore, phi, nx_wall, ny_wall = bte.build_pore_geometry(
        X, Y, Lx, Ly, xc, rfrac)

    # Linear initial guess
    T = np.zeros((Nx, Ny), dtype=float)
    for i in range(Nx):
        T[i, :] = TL + (TR - TL) * (x[i] / Lx)

    Ttilde = np.zeros((Nd, Nx, Ny), dtype=float)
    for n in range(Nd):
        Ttilde[n] = T.copy()

    residuals = []
    for it in range(1, max_iter + 1):
        T_old = T.copy()
        Ttilde_old = Ttilde.copy()

        for n, (mu_x, mu_y) in enumerate(dirs):
            Tn = Ttilde[n]
            i_range = range(0, Nx) if mu_x >= 0 else range(Nx - 1, -1, -1)
            j_range = range(0, Ny) if mu_y >= 0 else range(Ny - 1, -1, -1)

            for i in i_range:
                for j in j_range:
                    if is_pore[i, j]:
                        continue
                    # x-upwind
                    if mu_x > 0:
                        if i == 0:
                            T_up_x = TL
                        else:
                            i_up = i - 1
                            T_up_x = (bte.mixed_pore_bc_value(
                                        i, j, n, T_old, Ttilde_old, dirs,
                                        nx_wall, ny_wall, p_spec)
                                      if is_pore[i_up, j] else Tn[i_up, j])
                    elif mu_x < 0:
                        if i == Nx - 1:
                            T_up_x = TR
                        else:
                            i_up = i + 1
                            T_up_x = (bte.mixed_pore_bc_value(
                                        i, j, n, T_old, Ttilde_old, dirs,
                                        nx_wall, ny_wall, p_spec)
                                      if is_pore[i_up, j] else Tn[i_up, j])
                    else:
                        T_up_x = Tn[i, j]

                    # y-upwind (periodic)
                    if mu_y > 0:
                        j_up = j - 1 if j > 0 else Ny - 1
                    elif mu_y < 0:
                        j_up = j + 1 if j < Ny - 1 else 0
                    else:
                        j_up = j
                    T_up_y = (bte.mixed_pore_bc_value(
                                i, j, n, T_old, Ttilde_old, dirs,
                                nx_wall, ny_wall, p_spec)
                              if is_pore[i, j_up] else Tn[i, j_up])

                    # Local nonlinear MFP
                    if alpha == 0.0:
                        Lambda_loc = Lambda0
                    else:
                        Lambda_loc = max(Lambda0 * (1.0 + alpha * (T_old[i, j] - T_ref)),
                                         1e-8)

                    Tn[i, j] = bte.update_cell(mu_x, mu_y, dx, dy, Lambda_loc,
                                               T_old[i, j], T_up_x, T_up_y)

        # Angular closure on solid cells
        T_new = T_old.copy()
        for i in range(Nx):
            for j in range(Ny):
                if not is_pore[i, j]:
                    T_new[i, j] = float(np.sum(w * Ttilde[:, i, j]))

        diff = float(np.max(np.abs(T_new - T_old)))
        residuals.append(diff)
        T = T_new
        if diff < tol:
            break

    # Reduce moments
    qx = np.zeros((Nx, Ny), dtype=float)
    qy = np.zeros((Nx, Ny), dtype=float)
    for n, (mu_x, mu_y) in enumerate(dirs):
        qx += w[n] * mu_x * Ttilde[n]
        qy += w[n] * mu_y * Ttilde[n]

    solid = ~is_pore
    return dict(Jx=float(qx[solid].mean()),
                Jy=float(qy[solid].mean()),
                phi=phi, iters=it,
                final_res=residuals[-1] if residuals else np.nan)


def compute_case(bte, N, Nd, p_spec, xc, rfrac, alpha, max_iter, tol):
    problem = bte.build_problem(Nx=N, Ny=N, Nd=Nd)
    fwd = solve_with_qy(bte, problem, 1.0, 0.0, xc, rfrac, alpha,
                        p_spec, max_iter, tol)
    rev = solve_with_qy(bte, problem, 0.0, 1.0, xc, rfrac, alpha,
                        p_spec, max_iter, tol)
    absJf, absJr = abs(fwd["Jx"]), abs(rev["Jx"])
    R = (absJf - absJr) / min(absJf, absJr)
    qyF = abs(fwd["Jy"]) / max(abs(fwd["Jx"]), 1e-30)
    qyR = abs(rev["Jy"]) / max(abs(rev["Jx"]), 1e-30)
    return dict(
        Nx=N, Ny=N, Nd=Nd, xc=xc, rfrac=rfrac, alpha=alpha,
        p_spec=p_spec, phi=fwd["phi"],
        J_F=fwd["Jx"], J_R=rev["Jx"],
        R=R, absR_percent=100.0 * abs(R),
        Jy_F=fwd["Jy"], Jy_R=rev["Jy"],
        qy_over_qx_F=qyF, qy_over_qx_R=qyR,
        max_qy_over_qx=max(qyF, qyR),
        iters_F=fwd["iters"], iters_R=rev["iters"],
        final_res_F=fwd["final_res"], final_res_R=rev["final_res"],
    )


def import_solver_module(solver_path):
    """Dynamically load the BTE solver as a module to reuse its helpers."""
    spec = importlib.util.spec_from_file_location("bte_solver", str(solver_path))
    bte = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bte)
    return bte


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=str, default="bte_specularity_scan.py",
                    help="Path to the BTE solver Python file.")
    ap.add_argument("--outdir", type=str, default="all_results/revision_validation",
                    help="Where to write CSVs and the validation figure.")
    ap.add_argument("--grids", type=int, nargs="+", default=[81, 101, 121, 141],
                    help="Spatial resolutions to test.")
    ap.add_argument("--Nd", type=int, default=16,
                    help="Angular resolution (16 baseline, 32 for stronger check).")
    ap.add_argument("--xc", type=float, default=0.25)
    ap.add_argument("--rfrac", type=float, default=0.23)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--max_iter", type=int, default=6000)
    ap.add_argument("--tol", type=float, default=2e-6)
    args = ap.parse_args()

    solver_path = Path(args.solver).resolve()
    if not solver_path.exists():
        sys.exit(f"ERROR: solver not found at {solver_path}")
    bte = import_solver_module(solver_path)
    print(f"Imported solver from {solver_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run all (N, p_spec) combinations
    rows = []
    for N in args.grids:
        for p in (0.0, 1.0):
            print(f"\nRunning N={N}, Nd={args.Nd}, p_spec={p} ...")
            row = compute_case(bte, N=N, Nd=args.Nd, p_spec=p,
                               xc=args.xc, rfrac=args.rfrac, alpha=args.alpha,
                               max_iter=args.max_iter, tol=args.tol)
            rows.append(row)
            print(json.dumps({k: row[k] for k in
                              ("Nx", "p_spec", "J_F", "J_R", "absR_percent",
                               "qy_over_qx_F", "qy_over_qx_R",
                               "iters_F", "iters_R")},
                             indent=2))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "grid_p0_p1_with_transverse_flux.csv", index=False)

    # Build the summary (this is Table 5)
    summary_rows = []
    for N in args.grids:
        sub = df[df["Nx"] == N]
        r0 = float(sub[sub["p_spec"] == 0.0]["absR_percent"].iloc[0])
        r1 = float(sub[sub["p_spec"] == 1.0]["absR_percent"].iloc[0])
        S = 100.0 * (r0 - r1) / r0
        summary_rows.append({
            "Nx=Ny": N, "Nd": args.Nd,
            "|R| p=0 (%)": r0, "|R| p=1 (%)": r1,
            "relative suppression S (%)": S,
            "max |<qy>|/|<qx>|": float(sub["max_qy_over_qx"].max()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary_suppression_ratio_grid_qy.csv", index=False)

    print("\n=== Table 5 ===")
    print(summary.to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(summary["Nx=Ny"], summary["|R| p=0 (%)"], marker="o",
            label=r"$p_{\mathrm{spec}}=0$")
    ax.plot(summary["Nx=Ny"], summary["|R| p=1 (%)"], marker="s",
            label=r"$p_{\mathrm{spec}}=1$")
    ax.set_xlabel(r"$N_x=N_y$")
    ax.set_ylabel(r"$|R|$ (%)")
    ax.set_title("(a) Grid dependence of |R|")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(summary["Nx=Ny"], summary["relative suppression S (%)"], marker="o")
    ax.set_xlabel(r"$N_x=N_y$")
    ax.set_ylabel(r"$S = (|R|_{0}-|R|_{1})/|R|_{0}$ (%)")
    ax.set_title("(b) Relative suppression ratio")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.semilogy(summary["Nx=Ny"], summary["max |<qy>|/|<qx>|"], marker="o")
    ax.set_xlabel(r"$N_x=N_y$")
    ax.set_ylabel(r"max $|\langle q_y\rangle|/|\langle q_x\rangle|$")
    ax.set_title("(c) Transverse-flux check")
    ax.grid(True, alpha=0.3, which="both")

    fig.savefig(outdir / "Fig_revision_grid_suppression_qy.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "Fig_revision_grid_suppression_qy.pdf",
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs saved under {outdir}/")


if __name__ == "__main__":
    main()
