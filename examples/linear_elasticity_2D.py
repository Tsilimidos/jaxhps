"""
2D linear elasticity solver using jaxhps with Gauss-Seidel iteration.

Solves the Navier-Cauchy equations for homogeneous isotropic linear elasticity
(plane strain) on the domain [-1,1]^2.

Governing equations:
    (λ+2μ) u_x,xx + μ u_x,yy + (λ+μ) u_y,xy = -f_x
    (λ+μ) u_x,xy + μ u_y,xx + (λ+2μ) u_y,yy = -f_y

Strategy: Gauss-Seidel iteration treating the cross-derivative coupling as a
source term. The differential operators do not change between iterations, so
build_solver is called once per component (with source=None) to store the
source-independent operator matrices. Each iteration then only performs the
cheap upward+downward passes with the updated source.

    Iteration k:
        s_x^(k) = -f_x - (λ+μ) D_xy u_y^(k-1)
        u_x^(k) = solve( (λ+2μ)∂_xx + μ∂_yy, bc_x, source=s_x^(k) )

        s_y^(k) = -f_y - (λ+μ) D_xy u_x^(k)
        u_y^(k) = solve( μ∂_xx + (λ+2μ)∂_yy, bc_y, source=s_y^(k) )

The D_xy spectral differentiation matrix is applied leaf-wise:
    u_xy[leaf, :] = D_xy @ u[leaf, :]

Manufactured solution (for verification):
    u_x(x,y) = sin(π x) sin(π y)
    u_y(x,y) = cos(π x) cos(π y)

Body forces:
    f_x = π²(λ+3μ) sin(πx)sin(πy)
    f_y = π²(λ+3μ) cos(πx)cos(πy)

Usage:
    python examples/linear_elasticity_2D.py
    python examples/linear_elasticity_2D.py --p_vals 8 12 16 --l_vals 2 3
    python examples/linear_elasticity_2D.py --max_iter 20 --tol 1e-12
"""
import os
import time
import argparse
import logging

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.io import savemat

from jaxhps import (
    DiscretizationNode2D,
    build_solver,
    solve,
    PDEProblem,
    Domain,
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

jax.config.update("jax_default_device", jax.devices("cpu")[0])

# -------------------------------------------------------------------------
# Lamé constants
# -------------------------------------------------------------------------
LAMBDA = 1.0   # first Lamé parameter
MU = 1.0       # shear modulus

XMIN = -1.0
XMAX = 1.0
YMIN = -1.0
YMAX = 1.0

PI = jnp.pi


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D linear elasticity via Gauss-Seidel iteration with jaxhps."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/examples/linear_elasticity_2D",
    )
    parser.add_argument(
        "--p_vals", type=int, nargs="+", default=[3, 4, 6, 8, 12, 16]
    )
    parser.add_argument(
        "--l_vals", type=int, nargs="+", default=[2, 3, 4]
    )
    parser.add_argument(
        "--max_iter", type=int, default=30,
        help="Maximum number of Gauss-Seidel iterations."
    )
    parser.add_argument(
        "--tol", type=float, default=1e-14,
        help="L-inf convergence tolerance on the displacement update."
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# -------------------------------------------------------------------------
# Manufactured solution
# -------------------------------------------------------------------------

def ux_exact(x: jnp.ndarray) -> jnp.ndarray:
    """u_x(x,y) = sin(πx) sin(πy).  Input shape (..., 2)."""
    return jnp.sin(PI * x[..., 0]) * jnp.sin(PI * x[..., 1])


def uy_exact(x: jnp.ndarray) -> jnp.ndarray:
    """u_y(x,y) = cos(πx) cos(πy).  Input shape (..., 2)."""
    return jnp.cos(PI * x[..., 0]) * jnp.cos(PI * x[..., 1])


def fx_body(x: jnp.ndarray) -> jnp.ndarray:
    """Body force in x: f_x = 2μπ² sin(πx)sin(πy).

    From equilibrium: f_x = -(λ+2μ)u_x,xx - μ u_x,yy - (λ+μ)u_y,xy
                          = π²[(λ+2μ)+μ-(λ+μ)] sin = 2μπ² sin.
    """
    return 2 * MU * (PI**2) * ux_exact(x)


def fy_body(x: jnp.ndarray) -> jnp.ndarray:
    """Body force in y: f_y = 2μπ² cos(πx)cos(πy).

    From equilibrium: f_y = -(λ+μ)u_x,xy - μ u_y,xx - (λ+2μ)u_y,yy
                          = π²[-(λ+μ)+μ+(λ+2μ)] cos = 2μπ² cos.
    """
    return 2 * MU * (PI**2) * uy_exact(x)


# -------------------------------------------------------------------------
# Gauss-Seidel solver
# -------------------------------------------------------------------------

def gauss_seidel_elasticity(
    domain: Domain,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> tuple[jnp.ndarray, jnp.ndarray, list[float], float, float]:
    """
    Solve 2D linear elasticity on domain via Gauss-Seidel iteration.

    Calls build_solver once per component (source=None), then iterates
    using only the cheap solve (upward + downward pass) each step.

    Returns:
        u_x, u_y    : solution arrays of shape (n_leaves, p^2)
        residuals   : list of L-inf update norms per iteration
        t_build     : wall time (s) for the two build_solver calls
        t_solve     : wall time (s) for all GS iterations
    """
    ones = jnp.ones_like(domain.interior_points[..., 0])
    f_x = fx_body(domain.interior_points)
    f_y = fy_body(domain.interior_points)

    # When solve() is called with source=..., the up_pass promotes g_tilde_lst
    # to 3D (multi-source, nsrc=1). The down_pass then requires boundary_data
    # of shape (n_bdry, 1) instead of (n_bdry,), and returns (n_leaves, p^2, 1).
    bc_ux = jnp.expand_dims(ux_exact(domain.boundary_points), axis=-1)  # (n_bdry, 1)
    bc_uy = jnp.expand_dims(uy_exact(domain.boundary_points), axis=-1)  # (n_bdry, 1)

    # Build both operators once (no source — stores Phi, S_lst, D_inv_lst, BD_inv_lst)
    t0_build = time.perf_counter()
    pde_ux = PDEProblem(
        domain=domain,
        D_xx_coefficients=(LAMBDA + 2 * MU) * ones,
        D_yy_coefficients=MU * ones,
        # source=None triggers the nosource build path
    )
    build_solver(pde_ux)

    pde_uy = PDEProblem(
        domain=domain,
        D_xx_coefficients=MU * ones,
        D_yy_coefficients=(LAMBDA + 2 * MU) * ones,
    )
    build_solver(pde_uy)
    t_build = time.perf_counter() - t0_build

    # D_xy matrix: shape (p^2, p^2) — same for both since same domain/p
    D_xy = pde_ux.D_xy   # spectral mixed-derivative operator on each leaf

    # Initialize u_y^(0) = 0
    u_x = jnp.zeros_like(ones)
    u_y = jnp.zeros_like(ones)

    residuals = []
    t0_solve = time.perf_counter()

    for k in range(max_iter):
        u_x_old = u_x
        u_y_old = u_y

        # Step 1: compute coupling term from u_y^(k-1), apply D_xy leaf-wise
        #   u_y_xy[leaf, i] = sum_j D_xy[i,j] * u_y[leaf, j]
        u_y_xy = jnp.einsum("ij,lj->li", D_xy, u_y)
        s_x = -f_x - (LAMBDA + MU) * u_y_xy

        # Step 2: solve for u_x^(k) — cheap: only upward+downward pass
        u_x = solve(pde_ux, bc_ux, source=s_x)[..., 0]  # squeeze nsrc dim

        # Step 3: compute coupling term from u_x^(k)
        u_x_xy = jnp.einsum("ij,lj->li", D_xy, u_x)
        s_y = -f_y - (LAMBDA + MU) * u_x_xy

        # Step 4: solve for u_y^(k)
        u_y = solve(pde_uy, bc_uy, source=s_y)[..., 0]  # squeeze nsrc dim

        # Step 5: check convergence
        res = float(jnp.max(jnp.abs(u_x - u_x_old) + jnp.abs(u_y - u_y_old)))
        residuals.append(res)
        logging.debug("GS iter %i: update = %.4e", k + 1, res)

        if res < tol:
            logging.info("GS converged in %i iterations (update=%.4e)", k + 1, res)
            break
    else:
        logging.warning("GS did not converge in %i iterations", max_iter)

    t_solve = time.perf_counter() - t0_solve
    return u_x, u_y, residuals, t_build, t_solve


# -------------------------------------------------------------------------
# hp-convergence study
# -------------------------------------------------------------------------

def run_convergence(
    l_vals: jnp.ndarray,
    p_vals: jnp.ndarray,
    max_iter: int,
    tol: float,
) -> dict:
    """
    hp-convergence sweep.

    Returns a dict with keys:
        errors_ux, errors_uy  : (n_l, n_p) relative L-inf errors
        t_build, t_solve      : (n_l, n_p) wall times in seconds
        n_iters               : (n_l, n_p) GS iteration counts
        last_domain           : Domain for the last (l, p) case
        last_ux, last_uy      : final solutions on last_domain
    """
    nl, np_ = len(l_vals), len(p_vals)
    errors_ux = np.zeros((nl, np_))
    errors_uy = np.zeros((nl, np_))
    t_build   = np.zeros((nl, np_))
    t_solve   = np.zeros((nl, np_))
    n_iters   = np.zeros((nl, np_), dtype=int)

    last_domain = last_ux = last_uy = None

    for i, l in enumerate(l_vals):
        l = int(l)
        for j, p in enumerate(p_vals):
            p = int(p)

            root = DiscretizationNode2D(
                xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX
            )
            domain = Domain(p=p, q=max(1, p - 2), root=root, L=l)

            u_x, u_y, residuals, tb, ts = gauss_seidel_elasticity(
                domain, max_iter=max_iter, tol=tol
            )

            exact_ux = ux_exact(domain.interior_points)
            exact_uy = uy_exact(domain.interior_points)

            err_ux = float(
                jnp.max(jnp.abs(u_x - exact_ux)) / jnp.max(jnp.abs(exact_ux))
            )
            err_uy = float(
                jnp.max(jnp.abs(u_y - exact_uy)) / jnp.max(jnp.abs(exact_uy))
            )

            errors_ux[i, j] = err_ux
            errors_uy[i, j] = err_uy
            t_build[i, j]   = tb
            t_solve[i, j]   = ts
            n_iters[i, j]   = len(residuals)

            logging.info(
                "l=%i, p=%i, err_ux=%.4e, err_uy=%.4e, GS_iters=%i, "
                "t_build=%.2fs, t_solve=%.2fs",
                l, p, err_ux, err_uy, len(residuals), tb, ts,
            )

            last_domain, last_ux, last_uy = domain, u_x, u_y

    return dict(
        errors_ux=errors_ux,
        errors_uy=errors_uy,
        t_build=t_build,
        t_solve=t_solve,
        n_iters=n_iters,
        last_domain=last_domain,
        last_ux=last_ux,
        last_uy=last_uy,
    )


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]
COLORS  = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_runtimes(
    l_vals: np.ndarray,
    p_vals: np.ndarray,
    t_build: np.ndarray,
    t_solve: np.ndarray,
    output_dir: str,
) -> None:
    """Line plot of total wall time (build + solve) vs p, one line per L."""
    t_total = t_build + t_solve

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, l in enumerate(l_vals):
        ax.plot(
            p_vals, t_total[i],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=f"$L={l}$",
        )
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel("Total wall time (s)")
    ax.set_title("2D linear elasticity — total runtime")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fp = os.path.join(output_dir, "runtimes.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Runtime plot saved to: %s", fp)


def plot_iterations(
    l_vals: np.ndarray,
    p_vals: np.ndarray,
    n_iters: np.ndarray,
    output_dir: str,
) -> None:
    """Bar chart of GS iteration count vs p, grouped by l."""
    n_l, n_p = n_iters.shape
    x = np.arange(n_p)
    width = 0.8 / n_l

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, l in enumerate(l_vals):
        offset = (i - n_l / 2 + 0.5) * width
        ax.bar(
            x + offset, n_iters[i],
            width=width * 0.9,
            color=COLORS[i % len(COLORS)],
            label=f"$L={l}$",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in p_vals])
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel("Iterations")
    ax.set_title("2D linear elasticity — iteration count")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fp = os.path.join(output_dir, "iterations.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Iteration count plot saved to: %s", fp)


def plot_error_vs_p(
    l_vals: np.ndarray,
    p_vals: np.ndarray,
    errors_ux: np.ndarray,
    errors_uy: np.ndarray,
    output_dir: str,
) -> None:
    """Semilogy plot of relative L-inf error vs p, one line per L."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, errors, comp in zip(axes, [errors_ux, errors_uy], [r"$u_x$", r"$u_y$"]):
        for i, l in enumerate(l_vals):
            ax.semilogy(
                p_vals, errors[i],
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=f"$L={l}$",
            )
        ax.set_xlabel("Polynomial degree $p$")
        ax.set_ylabel("Relative $L^\\infty$ error")
        ax.set_title(f"Error in {comp}")
        ax.legend()
        ax.grid(True, which="both", linestyle=":")

    fig.suptitle("2D linear elasticity — hp-convergence")
    fig.tight_layout()
    fp = os.path.join(output_dir, "error_vs_p.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Error vs p plot saved to: %s", fp)


def plot_solution_contours(
    domain: Domain,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    output_dir: str,
    n_levels: int = 20,
) -> None:
    """
    Filled contour plots of u_x, u_y, |u_x - exact|, |u_y - exact|.
    Uses matplotlib tricontourf on the leaf Chebyshev points.
    """
    pts = np.array(domain.interior_points).reshape(-1, 2)
    x_pts, y_pts = pts[:, 0], pts[:, 1]

    ux_vals  = np.array(u_x).ravel()
    uy_vals  = np.array(u_y).ravel()
    ux_exact_vals = np.array(ux_exact(domain.interior_points)).ravel()
    uy_exact_vals = np.array(uy_exact(domain.interior_points)).ravel()
    err_ux = np.abs(ux_vals - ux_exact_vals)
    err_uy = np.abs(uy_vals - uy_exact_vals)

    triang = mtri.Triangulation(x_pts, y_pts)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    datasets = [
        (ux_vals,  r"$u_x$"),
        (uy_vals,  r"$u_y$"),
        (err_ux,   r"$|u_x - u_x^{\mathrm{exact}}|$"),
        (err_uy,   r"$|u_y - u_y^{\mathrm{exact}}|$"),
    ]
    cmaps = ["RdBu_r", "RdBu_r", "viridis", "viridis"]

    for ax, (vals, title), cmap in zip(axes.ravel(), datasets, cmaps):
        tcf = ax.tricontourf(triang, vals, levels=n_levels, cmap=cmap)
        fig.colorbar(tcf, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal")

    p = domain.p
    l = domain.L
    fig.suptitle(
        f"2D linear elasticity — solution contours  ($L={l}$, $p={p}$)"
    )
    fig.tight_layout()
    fp = os.path.join(output_dir, "solution_contours.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Solution contour plot saved to: %s", fp)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Output directory: %s", args.output_dir)
    logging.info("Lame constants: lambda=%.2f, mu=%.2f", LAMBDA, MU)
    logging.info("GS params: max_iter=%i, tol=%.2e", args.max_iter, args.tol)

    l_vals = np.array(args.l_vals)
    p_vals = np.array(args.p_vals)

    results = run_convergence(
        l_vals, p_vals, max_iter=args.max_iter, tol=args.tol
    )

    # --- Save all results to .mat ---
    fp = os.path.join(args.output_dir, "results.mat")
    savemat(
        fp,
        {
            "errors_ux":   results["errors_ux"],
            "errors_uy":   results["errors_uy"],
            "t_build":     results["t_build"],
            "t_solve":     results["t_solve"],
            "n_iters":     results["n_iters"],
            "l_vals":      l_vals,
            "p_vals":      p_vals,
            "lame_lambda": float(LAMBDA),
            "lame_mu":     float(MU),
        },
    )
    logging.info("Results saved to: %s", fp)

    # --- Plots ---
    plot_error_vs_p(
        l_vals, p_vals,
        results["errors_ux"], results["errors_uy"],
        args.output_dir,
    )
    plot_runtimes(
        l_vals, p_vals,
        results["t_build"], results["t_solve"],
        args.output_dir,
    )
    plot_iterations(
        l_vals, p_vals,
        results["n_iters"],
        args.output_dir,
    )
    plot_solution_contours(
        results["last_domain"],
        results["last_ux"],
        results["last_uy"],
        args.output_dir,
    )


if __name__ == "__main__":
    args = setup_args()
    logging.basicConfig(
        format="%(asctime)s:jaxhps: %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    main(args)
