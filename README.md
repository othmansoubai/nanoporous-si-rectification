# Pore-Wall Specularity Suppresses Thermal Rectification in Asymmetric Nanoporous Silicon

This repository contains the simulation code, post-processing scripts, and full
reproduction pipeline for the manuscript:

> **Pore-Wall Specularity Suppresses Thermal Rectification in Asymmetric
> Nanoporous Silicon: A Nonlinear Gray Phonon-BTE Study**
> Y. Abouelhanoune, O. Soubai, and M. Taibi
> *International Journal of Thermal Sciences* (under review).

A nonlinear gray phonon-Boltzmann transport equation (BTE) solver in two
dimensions, with a mixed diffuse-specular pore-wall boundary condition
controlled by a single specularity parameter `p_spec ∈ [0, 1]`. The headline
finding is a systematic ~17% suppression of thermal rectification as the
pore-wall changes from fully diffuse (`p_spec = 0`) to fully specular
(`p_spec = 1`) in an asymmetric circular-pore geometry, with the geometric
ranking of favorable designs preserved.

## What this repository does

Running a single command reproduces every simulation, every figure, and every
validation table in the manuscript:

```bash
python reproduce_paper.py
```

That command runs four stages: simulations (Stage 1), CSV aggregation
(Stage 2), publication figures (Stage 3), and the two validation analyses
that produce Tables 4 and 5 of the manuscript (Stage 4).

## Repository layout

```
.
├── README.md                           This file.
├── LICENSE                             MIT license.
├── requirements.txt                    Python dependencies.
├── CITATION.cff                        Machine-readable citation.
├── reproduce_paper.py                  Single-command end-to-end reproduction.
├── bte_specularity_scan.py             The BTE solver. Reads CLI parameters,
│                                         writes one summary.csv plus one .npz
│                                         field file per p_spec value.
├── generate_pub_figures.py             Loads the .npz field files and the
│                                         aggregated CSVs and produces all 9
│                                         publication figures (PDF + PNG).
├── analysis/
│   ├── grid_convergence_transverse_flux.py
│   │                                   Reproduces Table 5: grid convergence
│   │                                   of the suppression ratio and the
│   │                                   transverse-flux check.
│   └── gradient_orientation_analysis.py
│                                       Reproduces Table 4: temperature-
│                                       gradient orientation distributions
│                                       for the forward/reverse, p=0/p=1 cases.
└── all_results/                        Created at runtime; ignored by git.
    └── pub_figures/                    Created by the figure script.
```

## Quick start

```bash
# Clone and install
git clone https://github.com/<your-username>/nanoporous-si-rectification.git
cd nanoporous-si-rectification
pip install -r requirements.txt

# Quick smoke test (~30–60 min): baseline + p_spec scan only
python reproduce_paper.py --quick

# Full reproduction (~12–24 h on a single core)
python reproduce_paper.py
```

The full reproduction is idempotent: if you interrupt it and restart, it
skips any sweep whose `summary.csv` already exists.

## Data flow

The simulation pipeline is fully self-contained — no external phonon
database, DFT input, or AlmaBTE/OpenBTE dependency is required.

```
Parameters (xc, R/Lx, alpha, p_spec, Nx, Ny, Nd)
        │
        ▼
bte_specularity_scan.py                  (the solver)
        │
        │  ─► summary.csv    (scalar results: J_F, J_R, R, phi, ...)
        │  ─► <stem>.npz     (field arrays: Tf, Tr, mask, residuals, meta)
        │     stem = "xc{xc:.3f}_r{R:.3f}_a{alpha:.3f}_p{p_spec:.3f}"
        ▼
generate_pub_figures.py                  (post-processing)
        │
        ▼
pub_figures/Fig{1..9}.{png,pdf}          (9 publication figures)
analysis/.../*.csv  +  Fig_revision_*    (Tables 4 and 5 + revision figures)
```

The `.npz` field files contain the per-cell temperature solutions used by
Figure 5 and by the gradient-orientation analysis (Table 4). Every NPZ file
is a deterministic output of running the solver with a particular parameter
set; there is no external data input.

## Solver model and assumptions

- Two-dimensional Cartesian unit cell of unit normalized length `Lx = Ly = 1`.
- Single asymmetric circular pore at fractional position `xc/Lx`, fractional
  radius `R/Lx`, giving porosity `phi`.
- Gray (single-mode) phonon Boltzmann transport equation under the
  relaxation-time approximation.
- Reference dimensionless mean free path `Λ_0 = 0.1` (Knudsen-number-like
  parameter; e.g., `Lx = 100 nm` ⇒ `Λ_0 = 10 nm`).
- Nonlinear local mean free path: `Λ_loc = Λ_0 [1 + α (T - T_ref)]`.
- Mixed diffuse-specular pore-wall boundary condition with specularity
  parameter `p_spec ∈ [0, 1]`.
- Periodic boundaries in the transverse (`y`) direction (i.e., the model
  represents an idealized infinite transverse repetition of the unit cell).

See Section 2 of the manuscript for the full formulation.

## Reproducing individual results

If you do not want to run the full pipeline, you can call the solver
directly:

```bash
# Baseline: |R| versus p_spec at five specularity values
python bte_specularity_scan.py \
    --Nx 101 --Ny 101 --Nd 16 \
    --xc 0.25 --rfrac 0.23 --alpha 0.5 \
    --p_list 0.0,0.25,0.5,0.75,1.0 \
    --outdir all_results/results_pscan_fine

# A single (p_spec = 0) field run that produces an .npz file
python bte_specularity_scan.py \
    --Nx 101 --Ny 101 --Nd 16 \
    --xc 0.25 --rfrac 0.23 --alpha 0.5 \
    --p_spec 0.0 \
    --outdir all_results/results_fields_compare/p_0.0
```

After running the baseline cases, the two validation analyses can be run
standalone:

```bash
python analysis/grid_convergence_transverse_flux.py   # Table 5
python analysis/gradient_orientation_analysis.py      # Table 4
```

## Computational cost

The solver uses pure-Python nested loops over angular directions and grid
cells. This is research-grade code, not optimized for speed; expect roughly:

| Resolution      | Wall time per case      |
|-----------------|-------------------------|
| Nx=Ny=101, Nd=16 | ~3–10 min               |
| Nx=Ny=141, Nd=32 | ~30–90 min              |

A future speed-up (planned, see issues) is to JIT-compile the inner sweep
with Numba, which should give 50–100× speedup with minimal code change.

## Requirements

```
python >= 3.9
numpy
pandas
matplotlib
```

Pinned versions are in `requirements.txt`.

## Citation

If you use this code in your research, please cite the manuscript above and
this repository. The `CITATION.cff` file contains a machine-readable
citation that GitHub will render in the sidebar.

## License

MIT. See `LICENSE`.

## Authors

- **Younes Abouelhanoune** (corresponding author) — ENSAH Al-Hoceima,
  Abdelmalek Essaadi University, Morocco.
- **Othman Soubai** — LSA-EMAO, ENSAH Al-Hoceima, Abdelmalek Essaadi
  University, Morocco.
- **Mohammed Taibi** — LSA-EMAO, ENSAH Al-Hoceima, Abdelmalek Essaadi
  University, Morocco.
