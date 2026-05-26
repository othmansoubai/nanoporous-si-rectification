#!/usr/bin/env python3
import os, json, argparse, csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (4.8, 3.6),
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_problem(Nx=101, Ny=101, Nd=16):
    Lx = Ly = 1.0
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    Lambda0 = 0.1
    T_ref = 0.5

    theta = np.linspace(0, 2 * np.pi, Nd, endpoint=False)
    dirs = np.array([(float(np.cos(th)), float(np.sin(th))) for th in theta], dtype=float)
    w = np.full(Nd, 1.0 / Nd)
    return (Lx, Ly, Nx, Ny, x, y, dx, dy, X, Y, Lambda0, T_ref, Nd, dirs, w)


def build_pore_geometry(X, Y, Lx, Ly, x_c_frac, R_frac):
    if R_frac is None or R_frac <= 0.0:
        is_pore = np.zeros_like(X, dtype=bool)
        phi = 0.0
        nx_wall = np.zeros_like(X, dtype=float)
        ny_wall = np.zeros_like(X, dtype=float)
        return is_pore, phi, nx_wall, ny_wall

    x_c = x_c_frac * Lx
    y_c = 0.5 * Ly
    R_pore = R_frac * Lx
    dx = X - x_c
    dy = Y - y_c
    r = np.sqrt(dx**2 + dy**2)
    is_pore = (r <= R_pore)
    phi = float(is_pore.mean())

    # Normal from solid cell toward pore centerline (solid -> pore) on pore-adjacent solid cells.
    nx_wall = np.zeros_like(X, dtype=float)
    ny_wall = np.zeros_like(X, dtype=float)
    solid = ~is_pore
    Nx, Ny = X.shape
    for i in range(Nx):
        for j in range(Ny):
            if not solid[i, j]:
                continue
            touches_pore = False
            for ii, jj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if 0 <= ii < Nx and 0 <= jj < Ny and is_pore[ii, jj]:
                    touches_pore = True
                    break
            if not touches_pore:
                continue
            rr = r[i, j]
            if rr > 1e-14:
                nx_wall[i, j] = (x_c - X[i, j]) / rr
                ny_wall[i, j] = (y_c - Y[i, j]) / rr
    return is_pore, phi, nx_wall, ny_wall


def nearest_direction_index(s_target, dirs):
    dots = dirs @ s_target
    return int(np.argmax(dots))


def mixed_pore_bc_value(i, j, n_dir, T_old, Ttilde_old, dirs, nx_wall, ny_wall, p_spec):
    # p=0 -> exact Paper 1 diffuse rule: local solid temperature.
    T_diff = T_old[i, j]
    if p_spec <= 0.0:
        return T_diff

    nvec = np.array([nx_wall[i, j], ny_wall[i, j]], dtype=float)
    nrm = np.linalg.norm(nvec)
    if nrm < 1e-14:
        return T_diff
    nvec /= nrm

    s_in = dirs[n_dir]
    dot_in = float(np.dot(s_in, nvec))

    # Only incoming-to-solid directions should use reflected info.
    # If classification is ambiguous, fall back to diffuse.
    if dot_in >= 0.0:
        return T_diff

    s_ref = s_in - 2.0 * dot_in * nvec
    m_ref = nearest_direction_index(s_ref, dirs)
    T_spec = float(Ttilde_old[m_ref, i, j])
    return float(p_spec * T_spec + (1.0 - p_spec) * T_diff)


def update_cell(mu_x, mu_y, dx, dy, Lambda_loc, T_center, T_up_x, T_up_y):
    s = 1.0 / Lambda_loc
    ax = abs(mu_x) / dx
    ay = abs(mu_y) / dy
    rhs = ax * T_up_x + ay * T_up_y + s * T_center
    return rhs / (ax + ay + s)


def solve_2D_pore(problem, TL, TR, x_c_frac, R_frac, alpha, p_spec=0.0, max_iter=6000, tol=2e-6):
    (Lx, Ly, Nx, Ny, x, y, dx, dy, X, Y, Lambda0, T_ref, Nd, dirs, w) = problem
    is_pore, phi, nx_wall, ny_wall = build_pore_geometry(X, Y, Lx, Ly, x_c_frac, R_frac)

    T = np.zeros((Nx, Ny))
    for i in range(Nx):
        T[i, :] = TL + (TR - TL) * (x[i] / Lx)

    Ttilde = np.zeros((Nd, Nx, Ny))
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
                            if is_pore[i_up, j]:
                                T_up_x = mixed_pore_bc_value(i, j, n, T_old, Ttilde_old, dirs, nx_wall, ny_wall, p_spec)
                            else:
                                T_up_x = Tn[i_up, j]
                    elif mu_x < 0:
                        if i == Nx - 1:
                            T_up_x = TR
                        else:
                            i_up = i + 1
                            if is_pore[i_up, j]:
                                T_up_x = mixed_pore_bc_value(i, j, n, T_old, Ttilde_old, dirs, nx_wall, ny_wall, p_spec)
                            else:
                                T_up_x = Tn[i_up, j]
                    else:
                        T_up_x = Tn[i, j]

                    # y-upwind (periodic)
                    if mu_y > 0:
                        j_up = j - 1 if j > 0 else Ny - 1
                    elif mu_y < 0:
                        j_up = j + 1 if j < Ny - 1 else 0
                    else:
                        j_up = j

                    if is_pore[i, j_up]:
                        T_up_y = mixed_pore_bc_value(i, j, n, T_old, Ttilde_old, dirs, nx_wall, ny_wall, p_spec)
                    else:
                        T_up_y = Tn[i, j_up]

                    if alpha == 0.0:
                        Lambda_loc = Lambda0
                    else:
                        Lambda_loc = Lambda0 * (1.0 + alpha * (T_old[i, j] - T_ref))
                        Lambda_loc = max(Lambda_loc, 1e-8)

                    Tn[i, j] = update_cell(mu_x, mu_y, dx, dy, Lambda_loc, T_old[i, j], T_up_x, T_up_y)

        T_new = T_old.copy()
        solid = ~is_pore
        for i in range(Nx):
            for j in range(Ny):
                if solid[i, j]:
                    T_new[i, j] = float(np.sum(w * Ttilde[:, i, j]))

        diff = float(np.max(np.abs(T_new - T_old)))
        residuals.append(diff)
        T = T_new
        if diff < tol:
            break

    qx = np.zeros((Nx, Ny))
    for n, (mu_x, _) in enumerate(dirs):
        qx += w[n] * mu_x * Ttilde[n]

    J = float(np.mean(qx[~is_pore]))
    gradT = (TR - TL) / Lx
    k_rel = float(-J / gradT) if gradT != 0 else np.nan

    return T, J, k_rel, is_pore, phi, it, residuals


def compute_rectification(problem, x_c_frac, R_frac, alpha, p_spec=0.0, max_iter=6000, tol=2e-6):
    Tf, Jf, kf, mask, phi, itf, resf = solve_2D_pore(problem, 1.0, 0.0, x_c_frac, R_frac, alpha, p_spec, max_iter, tol)
    Tr, Jr, kr, _, _, itr, resr = solve_2D_pore(problem, 0.0, 1.0, x_c_frac, R_frac, alpha, p_spec, max_iter, tol)
    absJf, absJr = abs(Jf), abs(Jr)
    R = (absJf - absJr) / min(absJf, absJr)
    return Tf, Tr, Jf, Jr, R, phi, mask, itf, itr, resf, resr


def save_summary_row(path, row, write_header=False):
    header = list(row.keys())
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_case(args, p_spec, outdir):
    problem = build_problem(Nx=args.Nx, Ny=args.Ny, Nd=args.Nd)
    Tf, Tr, Jf, Jr, R, phi, mask, itf, itr, resf, resr = compute_rectification(
        problem,
        x_c_frac=args.xc,
        R_frac=args.rfrac,
        alpha=args.alpha,
        p_spec=p_spec,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    absR = 100.0 * abs(R)
    result = {
        'Nx': args.Nx, 'Ny': args.Ny, 'Nd': args.Nd,
        'xc': args.xc, 'rfrac': args.rfrac, 'alpha': args.alpha,
        'p_spec': p_spec, 'phi': phi,
        'J_F': Jf, 'J_R': Jr, 'R': R, 'absR_percent': absR,
        'iters_F': itf, 'iters_R': itr,
        'final_res_F': resf[-1] if resf else None,
        'final_res_R': resr[-1] if resr else None,
    }

    stem = f"xc{args.xc:.3f}_r{args.rfrac:.3f}_a{args.alpha:.3f}_p{p_spec:.3f}"
    np.savez_compressed(
        os.path.join(outdir, f"{stem}.npz"),
        Tf=Tf, Tr=Tr, mask=mask,
        resF=np.array(resf), resR=np.array(resr),
        meta=json.dumps(result),
    )
    return result


def plot_R_vs_p(results, outdir):
    ps = [r['p_spec'] for r in results]
    Rs = [r['R'] for r in results]
    fig, ax = plt.subplots()
    ax.plot(ps, Rs, marker='o')
    ax.axhline(0.0, linewidth=1.0, color='0.5')
    ax.set_xlabel('p_spec')
    ax.set_ylabel('Rectification R')
    ax.set_title('R vs pore-wall specularity')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'R_vs_p_spec.png'), dpi=250, bbox_inches='tight')
    fig.savefig(os.path.join(outdir, 'R_vs_p_spec.pdf'), dpi=250, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Paper 2 continuation from your Paper 1 code: mixed pore-wall specularity scan.')
    ap.add_argument('--Nx', type=int, default=101)
    ap.add_argument('--Ny', type=int, default=101)
    ap.add_argument('--Nd', type=int, default=16)
    ap.add_argument('--xc', type=float, default=0.25)
    ap.add_argument('--rfrac', type=float, default=0.23)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--p_spec', type=float, default=0.0)
    ap.add_argument('--p_list', type=str, default='')
    ap.add_argument('--max_iter', type=int, default=6000)
    ap.add_argument('--tol', type=float, default=2e-6)
    ap.add_argument('--outdir', type=str, default='paper2_results')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    summary_csv = os.path.join(args.outdir, 'summary.csv')
    if os.path.exists(summary_csv):
        os.remove(summary_csv)

    if args.p_list.strip():
        p_values = [float(s) for s in args.p_list.split(',')]
    else:
        p_values = [args.p_spec]

    results = []
    for idx, p in enumerate(p_values):
        print(f"Running case {idx+1}/{len(p_values)}: p_spec={p:.3f}")
        res = run_case(args, p, args.outdir)
        save_summary_row(summary_csv, res, write_header=(idx == 0))
        results.append(res)
        print(json.dumps(res, indent=2))

    with open(os.path.join(args.outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    if len(results) > 1:
        plot_R_vs_p(results, args.outdir)

    print(f"Saved results to: {args.outdir}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == '__main__':
    main()
