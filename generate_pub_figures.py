#!/usr/bin/env python3
"""
Publication-quality figures for:
"Pore-Wall Specularity Suppresses Thermal Rectification
 in Asymmetric Nanoporous Silicon"

Usage:
    python generate_pub_figures_v6.py

    For Figure 5 (temperature fields), place your npz data files at:
      all_results/results_fields_compare/p_0.0/xc0.250_r0.230_a0.500_p0.000.npz
      all_results/results_fields_compare/p_1.0/xc0.250_r0.230_a0.500_p1.000.npz
    (relative to this script). If not found, Fig 5 is skipped with a warning.

Outputs PDF + PNG for each figure in ./pub_figures/

Figure plan (9 figures):
    Fig 1: Geometry schematic
    Fig 2: Sanity checks
    Fig 3: Baseline |R| vs p_spec
    Fig 4: Flux-based mechanism
    Fig 5: Temperature field maps (NEW — replaces old design map)
    Fig 6: |R| vs pore position
    Fig 7: |R| vs porosity
    Fig 8: |R| vs alpha
    Fig 9: Convergence and hi-fi confirmation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import os

# ═══════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════

COL1 = 3.4
COL2 = 7.0

C_BLUE   = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN  = "#009E73"
C_GRAY   = "#999999"

PSPEC_STYLES = {
    0.0: dict(color=C_BLUE,   marker='o', ms=5,
              label=r'$p_{\mathrm{spec}}=0$'),
    0.5: dict(color=C_ORANGE, marker='s', ms=5,
              label=r'$p_{\mathrm{spec}}=0.5$'),
    1.0: dict(color=C_GREEN,  marker='^', ms=5,
              label=r'$p_{\mathrm{spec}}=1$'),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.2, "lines.markersize": 5,
    "axes.linewidth": 0.6, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": "0.80", "legend.handlelength": 1.8,
    "legend.borderpad": 0.4, "legend.handletextpad": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.constrained_layout.use": True,
})

OUTDIR = "pub_figures"
os.makedirs(OUTDIR, exist_ok=True)

# Path to field data (relative to working directory)
# Works in both scripts and Jupyter/Colab notebooks
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()  # Jupyter / Colab fallback
FIELD_P0 = os.path.join(SCRIPT_DIR,
    "all_results", "results_fields_compare", "p_0.0",
    "xc0.250_r0.230_a0.500_p0.000.npz")
FIELD_P1 = os.path.join(SCRIPT_DIR,
    "all_results", "results_fields_compare", "p_1.0",
    "xc0.250_r0.230_a0.500_p1.000.npz")


def savefig(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=300)
    plt.close(fig)
    print(f"  saved {name}")


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

baseline_p    = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
baseline_JF   = np.array([0.035180, 0.035337, 0.035509, 0.035693, 0.035888])
baseline_JR   = np.array([-0.035809, -0.035941, -0.036090, -0.036250, -0.036422])
baseline_absR = np.array([1.7895, 1.7101, 1.6339, 1.5596, 1.4861])

sanity_center_lin_p = [0.0, 0.5, 1.0]
sanity_center_lin_R = [6.84e-13, 6.98e-13, 6.93e-13]
sanity_offcen_lin_p = [0.0, 0.5, 1.0]
sanity_offcen_lin_R = [8.40e-13, 8.13e-13, 8.05e-13]
sanity_center_nl_p  = [0.0, 0.5, 1.0]
sanity_center_nl_R  = [0.0107, 0.0027, 0.0027]

xc_vals = [0.25, 0.30, 0.35, 0.40, 0.45]
xc_R = {
    0.0: [1.5036, 1.1761, 0.8753, 0.5703, 0.2860],
    0.5: [1.3782, 1.0740, 0.7913, 0.5191, 0.2597],
    1.0: [1.2563, 0.9748, 0.7116, 0.4705, 0.2341],
}

phi_vals = [0.0308, 0.0691, 0.1231, 0.1620]
phi_R = {
    0.0: [0.7058, 1.0770, 1.5036, 1.7895],
    0.5: [0.6702, 0.9978, 1.3782, 1.6339],
    1.0: [0.6274, 0.9142, 1.2563, 1.4861],
}

alpha_vals = [0.0, 0.1, 0.2, 0.3, 0.5]
alpha_R = {
    0.0: [8.40e-13, 0.3575, 0.7157, 1.0742, 1.7895],
    0.5: [8.13e-13, 0.3261, 0.6529, 0.9801, 1.6339],
    1.0: [8.05e-13, 0.2964, 0.5934, 0.8910, 1.4861],
}

ang_Nd = [16, 24, 32]
ang_R  = [1.4861, 1.4898, 1.4991]
grid_N = [81, 101, 121, 141]
grid_R = [1.5464, 1.4996, 1.4084, 1.3819]
hifi_labels = [r'$(0.25,\,0.23)$', r'$(0.35,\,0.20)$']
hifi_p0 = [1.6845, 0.8174]
hifi_p1 = [1.3819, 0.6760]


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Geometry schematic
# ═══════════════════════════════════════════════════════════════
def fig1_geometry():
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.95))

    rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='k', lw=1.2)
    ax.add_patch(rect)

    xc, yc, R = 0.25, 0.5, 0.20
    pore = Circle((xc, yc), R, facecolor='#E8E8E8', edgecolor='k',
                  lw=0.8, zorder=3)
    ax.add_patch(pore)

    ax.plot([xc, xc + R * 0.71], [yc, yc + R * 0.71], '-', color='0.4', lw=0.5)
    ax.annotate(r'$R_{\mathrm{pore}}$', xy=(xc, yc + R), fontsize=7.5,
                xytext=(xc + 0.03, yc + R + 0.10),
                arrowprops=dict(arrowstyle='->', lw=0.6, color='0.3'),
                color='0.2')

    ax.annotate('', xy=(xc, -0.08), xytext=(0, -0.08),
                arrowprops=dict(arrowstyle='<->', lw=0.5, color='0.3'))
    ax.text(xc / 2, -0.13, r'$x_c$', ha='center', va='top',
            fontsize=7.5, color='0.2')

    ax.annotate('', xy=(1, -0.20), xytext=(0, -0.20),
                arrowprops=dict(arrowstyle='<->', lw=0.5, color='0.3'))
    ax.text(0.5, -0.25, r'$L_x$', ha='center', va='top',
            fontsize=7.5, color='0.2')

    ax.annotate('', xy=(-0.10, 1), xytext=(-0.10, 0),
                arrowprops=dict(arrowstyle='<->', lw=0.5, color='0.3'))
    ax.text(-0.15, 0.5, r'$L_y$', ha='right', va='center',
            fontsize=7.5, color='0.2', rotation=90)

    ax.text(-0.04, 0.5, r'$T_L$', ha='right', va='center',
            fontsize=9, fontweight='bold', color=C_BLUE)
    ax.text(1.04, 0.5, r'$T_R$', ha='left', va='center',
            fontsize=9, fontweight='bold', color=C_ORANGE)

    y_arr = 0.92
    ax.annotate('', xy=(0.78, y_arr), xytext=(0.38, y_arr),
                arrowprops=dict(arrowstyle='->', lw=1.0, color=C_BLUE))
    ax.text(0.58, y_arr + 0.04, 'Forward', ha='center',
            fontsize=6.5, color=C_BLUE, style='italic')
    ax.annotate('', xy=(0.38, y_arr - 0.08), xytext=(0.78, y_arr - 0.08),
                arrowprops=dict(arrowstyle='->', lw=1.0, color=C_ORANGE,
                                linestyle='--'))
    ax.text(0.58, y_arr - 0.15, 'Reverse', ha='center',
            fontsize=6.5, color=C_ORANGE, style='italic')

    ax.text(0.5, 1.06, 'Periodic', ha='center', fontsize=7, color=C_GRAY)
    ax.text(0.5, -0.35, 'Periodic', ha='center', fontsize=7, color=C_GRAY)

    ax.annotate('', xy=(0.95, 0.5), xytext=(0.85, 0.5),
                arrowprops=dict(arrowstyle='->', lw=0.6, color='0.3'))
    ax.text(0.90, 0.54, r'$q_x$', ha='center', fontsize=8, color='0.3')

    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9)
    ax.text(0.75, 0.15,
            (r'$p_{\mathrm{spec}}=0$: diffuse' '\n'
             r'$p_{\mathrm{spec}}=1$: specular' '\n'
             r'$0<p_{\mathrm{spec}}<1$: mixed'),
            fontsize=5.5, va='center', ha='center',
            bbox=bbox_props, color='0.2')

    ax.set_xlim(-0.22, 1.15)
    ax.set_ylim(-0.40, 1.15)
    ax.set_aspect('equal')
    ax.axis('off')
    savefig(fig, "Fig1_geometry")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Sanity checks
# ═══════════════════════════════════════════════════════════════
def fig2_sanity():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.2))
    box_kw = dict(facecolor='white', edgecolor='0.85',
                  boxstyle='round,pad=0.3')

    ax = axes[0]
    ax.scatter(sanity_center_lin_p, sanity_center_lin_R,
               c=C_BLUE, s=35, zorder=5, marker='o',
               edgecolors='k', linewidths=0.3)
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.set_xlabel(r'$p_{\mathrm{spec}}$')
    ax.set_title(r'(a) Centered, $\alpha=0$', fontsize=8)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-13, -13))
    ax.set_xlim(-0.15, 1.15)
    ax.text(0.97, 0.05, r'$|R| \sim 10^{-13}\%$' '\n(machine zero)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.5, color=C_GRAY, bbox=box_kw)

    ax = axes[1]
    ax.scatter(sanity_offcen_lin_p, sanity_offcen_lin_R,
               c=C_BLUE, s=35, zorder=5, marker='o',
               edgecolors='k', linewidths=0.3)
    ax.set_xlabel(r'$p_{\mathrm{spec}}$')
    ax.set_title(r'(b) Off-centered, $\alpha=0$', fontsize=8)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-13, -13))
    ax.set_xlim(-0.15, 1.15)
    ax.text(0.03, 0.05, r'$|R| \sim 10^{-13}\%$' '\n(machine zero)',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=6.5, color=C_GRAY, bbox=box_kw)

    ax = axes[2]
    ax.scatter(sanity_center_nl_p, sanity_center_nl_R,
               c=C_BLUE, s=35, zorder=5, marker='o',
               edgecolors='k', linewidths=0.3)
    ax.plot(sanity_center_nl_p, sanity_center_nl_R,
            color=C_BLUE, lw=0.7, alpha=0.4)
    ax.set_xlabel(r'$p_{\mathrm{spec}}$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.set_title(r'(c) Centered, $\alpha=0.5$', fontsize=8)
    ax.set_xlim(-0.15, 1.15)
    ax.text(0.97, 0.55, r'$|R| < 0.011\%$' '\n(negligible)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=6.5, color=C_GRAY, bbox=box_kw)

    savefig(fig, "Fig2_sanity_checks")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Baseline |R| vs p_spec
# ═══════════════════════════════════════════════════════════════
def fig3_baseline():
    fig, ax = plt.subplots(figsize=(COL1, 2.6))

    ax.plot(baseline_p, baseline_absR, '-o', color=C_BLUE,
            ms=5, mfc='white', mew=1.2, zorder=5)

    ax.set_xlabel(r'Specularity $p_{\mathrm{spec}}$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.42, 1.86)

    bbox_props = dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.75")
    ax.text(0.97, 0.97,
            (r'$x_c/L_x = 0.25$' '\n'
             r'$R_{\mathrm{pore}}/L_x = 0.23$' '\n'
             r'$\phi \approx 0.162$' '\n'
             r'$\alpha = 0.50$'),
            transform=ax.transAxes, fontsize=6.5,
            va='top', ha='right', bbox=bbox_props)

    ax.annotate(f'{baseline_absR[0]:.2f}%', xy=(0, baseline_absR[0]),
                xytext=(0.15, 1.84), fontsize=6.5, color='0.4',
                arrowprops=dict(arrowstyle='->', lw=0.4, color='0.6'))
    ax.annotate(f'{baseline_absR[-1]:.2f}%', xy=(1, baseline_absR[-1]),
                xytext=(0.82, 1.44), fontsize=6.5, color='0.4',
                arrowprops=dict(arrowstyle='->', lw=0.4, color='0.6'))

    savefig(fig, "Fig3_baseline_R_vs_pspec")


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Flux-based mechanism
# ═══════════════════════════════════════════════════════════════
def fig4_flux_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3))

    absJF  = baseline_JF
    absJR  = np.abs(baseline_JR)
    deltaJ = absJR - absJF

    ax = axes[0]
    ax.plot(baseline_p, absJF, '-o', color=C_BLUE, ms=4.5,
            mfc='white', mew=1.0)
    ax.plot(baseline_p, absJR, '-s', color=C_ORANGE, ms=4.5,
            mfc='white', mew=1.0)
    ax.set_xlabel(r'$p_{\mathrm{spec}}$')
    ax.set_ylabel('Flux magnitude')
    ax.set_title(r'(a) $|J_F|$ and $|J_R|$', fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    h_JF = Line2D([], [], color=C_BLUE, marker='o', ms=5,
                  mfc='white', mew=1.0, linestyle='-', label=r'$|J_F|$')
    h_JR = Line2D([], [], color=C_ORANGE, marker='s', ms=5,
                  mfc='white', mew=1.0, linestyle='-', label=r'$|J_R|$')
    ax.legend(handles=[h_JF, h_JR], loc='lower right', fontsize=6.5,
              handlelength=2.5)

    ax = axes[1]
    ax.plot(baseline_p, deltaJ, '-D', color=C_GREEN, ms=4.5,
            mfc='white', mew=1.0)
    ax.set_xlabel(r'$p_{\mathrm{spec}}$')
    ax.set_ylabel(r'$\Delta J = |J_R| - |J_F|$')
    ax.set_title('(b) Directional flux contrast', fontsize=8)
    ax.set_xlim(-0.05, 1.05)

    ax = axes[2]
    ax.plot(baseline_p, baseline_absR, '-o', color=C_BLUE, ms=4.5,
            mfc='white', mew=1.0)
    ax.set_xlabel(r'$p_{\mathrm{spec}}$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.set_title('(c) Rectification magnitude', fontsize=8)
    ax.set_xlim(-0.05, 1.05)

    savefig(fig, "Fig4_flux_mechanism")


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Temperature field maps  (NEW)
#   2×2: rows = diffuse / specular,  cols = forward / reverse
# ═══════════════════════════════════════════════════════════════
def fig5_temperature_fields():
    """
    Load 2D temperature fields from npz and plot as colormaps.
    Pore region is masked (shown in gray).
    """
    if not (os.path.isfile(FIELD_P0) and os.path.isfile(FIELD_P1)):
        print("  ⚠ SKIPPED Fig5_temperature_fields — npz files not found.")
        print(f"    Expected:\n      {FIELD_P0}\n      {FIELD_P1}")
        print("    Adjust FIELD_P0 / FIELD_P1 paths at the top of this script.")
        return

    d0 = np.load(FIELD_P0, allow_pickle=True)
    d1 = np.load(FIELD_P1, allow_pickle=True)

    Tf_p0, Tr_p0 = d0['Tf'], d0['Tr']
    Tf_p1, Tr_p1 = d1['Tf'], d1['Tr']
    mask = d0['mask'].astype(bool)   # True inside pore

    Nx, Ny = Tf_p0.shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)

    # Mask pore cells with NaN so they appear as a hole
    def apply_mask(T, m):
        T_plot = T.copy().astype(float)
        T_plot[m] = np.nan
        return T_plot

    fig = plt.figure(figsize=(COL2, 6.2))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.04],
                           wspace=0.01, hspace=0.12, figure=fig)

    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad(color='0.85')

    panels = [
        (gs[0, 0], apply_mask(Tf_p0, mask), r'(a) Forward, $p_{\mathrm{spec}}=0$'),
        (gs[0, 1], apply_mask(Tr_p0, mask), r'(b) Reverse, $p_{\mathrm{spec}}=0$'),
        (gs[1, 0], apply_mask(Tf_p1, mask), r'(c) Forward, $p_{\mathrm{spec}}=1$'),
        (gs[1, 1], apply_mask(Tr_p1, mask), r'(d) Reverse, $p_{\mathrm{spec}}=1$'),
    ]

    im = None
    for idx, (gs_pos, T_plot, title) in enumerate(panels):
        ax = fig.add_subplot(gs_pos)
        im = ax.pcolormesh(x, y, T_plot.T, cmap=cmap,
                           vmin=0, vmax=1, shading='auto', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=8)

        # Only bottom row gets x-axis label
        if idx >= 2:
            ax.set_xlabel(r'$x / L_x$')

        # Only left column gets y-labels
        if idx % 2 == 0:
            ax.set_ylabel(r'$y / L_y$')
        else:
            ax.set_yticklabels([])

        circle = Circle((0.25, 0.5), 0.23, fill=False,
                         edgecolor='k', lw=0.6)
        ax.add_patch(circle)

    # Colorbar in dedicated narrow column
    cbar_ax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(r'Normalized temperature $T$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    savefig(fig, "Fig5_temperature_fields")


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: |R| vs pore position  (was Fig 5)
# ═══════════════════════════════════════════════════════════════
def fig6_xc_scan():
    fig, ax = plt.subplots(figsize=(COL1, 2.8))

    for p in [0.0, 0.5, 1.0]:
        st = PSPEC_STYLES[p]
        ax.plot(xc_vals, xc_R[p], '-', color=st['color'],
                marker=st['marker'], ms=st['ms'], mfc='white', mew=1.0,
                label=st['label'])

    ax.set_xlabel(r'Pore position $x_c/L_x$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.legend(loc='upper right', fontsize=6.5)
    ax.set_xlim(0.23, 0.47)

    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.75")
    ax.text(0.03, 0.03,
            r'$R_{\mathrm{pore}}/L_x=0.20$, $\alpha=0.5$',
            transform=ax.transAxes, fontsize=6.5, va='bottom',
            bbox=bbox_props)

    savefig(fig, "Fig6_R_vs_xc")


# ═══════════════════════════════════════════════════════════════
# FIGURE 7: |R| vs porosity  (was Fig 6)
# ═══════════════════════════════════════════════════════════════
def fig7_porosity():
    fig, ax = plt.subplots(figsize=(COL1, 2.8))

    for p in [0.0, 0.5, 1.0]:
        st = PSPEC_STYLES[p]
        ax.plot(phi_vals, phi_R[p], '-', color=st['color'],
                marker=st['marker'], ms=st['ms'], mfc='white', mew=1.0,
                label=st['label'])

    ax.set_xlabel(r'Porosity $\phi$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.legend(loc='upper left', fontsize=6.5)

    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.75")
    ax.text(0.97, 0.03, r'$x_c/L_x=0.25$, $\alpha=0.5$',
            transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right',
            bbox=bbox_props)

    savefig(fig, "Fig7_R_vs_porosity")


# ═══════════════════════════════════════════════════════════════
# FIGURE 8: |R| vs alpha  (was Fig 7)
# ═══════════════════════════════════════════════════════════════
def fig8_alpha():
    fig, ax = plt.subplots(figsize=(COL1, 2.8))

    for p in [0.0, 0.5, 1.0]:
        st = PSPEC_STYLES[p]
        ax.plot(alpha_vals[1:], alpha_R[p][1:], '-', color=st['color'],
                marker=st['marker'], ms=st['ms'], mfc='white', mew=1.0,
                label=st['label'])

    ax.axvspan(-0.02, 0.05, color='0.93', zorder=0)
    ax.text(0.025, 0.50,
            r'$\alpha\!=\!0$:' '\n' r'$|R|\!\approx\! 0$',
            fontsize=5.5, ha='center', color='0.5', style='italic')

    ax.set_xlabel(r'Nonlinear strength $\alpha$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.legend(loc='center left', bbox_to_anchor=(0.02, 0.72), fontsize=6.5)
    ax.set_xlim(-0.02, 0.55)

    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.75")
    ax.text(0.97, 0.03,
            r'$x_c/L_x=0.25$, $R_{\mathrm{pore}}/L_x=0.23$',
            transform=ax.transAxes, fontsize=6.5, va='bottom', ha='right',
            bbox=bbox_props)

    savefig(fig, "Fig8_R_vs_alpha")


# ═══════════════════════════════════════════════════════════════
# FIGURE 9: Convergence and hi-fi
# ═══════════════════════════════════════════════════════════════
def fig9_convergence():
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3))

    ax = axes[0]
    ax.plot(ang_Nd, ang_R, '-o', color=C_BLUE, ms=5, mfc='white', mew=1.0)
    ax.set_xlabel(r'$N_d$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.set_title('(a) Angular convergence', fontsize=8)
    ax.set_xticks(ang_Nd)
    ax.set_ylim(1.48, 1.51)

    ax = axes[1]
    ax.plot(grid_N, grid_R, '-s', color=C_ORANGE, ms=5, mfc='white', mew=1.0)
    ax.set_xlabel(r'$N_x = N_y$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.set_title('(b) Grid sensitivity', fontsize=8)

    ax = axes[2]
    x_pos = np.arange(len(hifi_labels))
    w = 0.30
    bars_p0 = ax.bar(x_pos - w / 2, hifi_p0, w, color=C_BLUE,
                      edgecolor='k', lw=0.4,
                      label=r'$p_{\mathrm{spec}}=0$', zorder=5)
    bars_p1 = ax.bar(x_pos + w / 2, hifi_p1, w, color=C_ORANGE,
                      edgecolor='k', lw=0.4,
                      label=r'$p_{\mathrm{spec}}=1$', zorder=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hifi_labels, fontsize=6.5)
    ax.set_xlabel(r'$(x_c/L_x,\; R_{\mathrm{pore}}/L_x)$')
    ax.set_ylabel(r'$|R|\;(\%)$')
    ax.set_title('(c) Higher-fidelity confirmation', fontsize=8)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_ylim(0, 2.0)
    for bar in list(bars_p0) + list(bars_p1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f'{bar.get_height():.2f}',
                ha='center', fontsize=5.5, color='0.3')

    savefig(fig, "Fig9_convergence_hifi")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication-quality figures (v6)...")
    print(f"Output directory: {os.path.abspath(OUTDIR)}/\n")

    fig1_geometry()
    fig2_sanity()
    fig3_baseline()
    fig4_flux_mechanism()
    fig5_temperature_fields()
    fig6_xc_scan()
    fig7_porosity()
    fig8_alpha()
    fig9_convergence()

    print(f"\nDone! 9 figures saved (PDF + PNG each).")
    print(f"Use the PDFs in your LaTeX manuscript.")