"""
2D linear elasticity — Mode I crack-tip singularity benchmark.

Tests hp-convergence of jaxhps on a problem whose exact solution has a
sqrt(r) corner singularity (Williams expansion).  Unlike the smooth
manufactured-solution case, the singularity limits polynomial approximation
to *algebraic* rather than spectral convergence.

Governing equations (plane strain, zero body force):
    (λ+2μ) u_x,xx + μ u_x,yy + (λ+μ) u_y,xy = 0
    (λ+μ) u_x,xy + μ u_y,xx + (λ+2μ) u_y,yy = 0

Exact solution: Mode I Williams expansion (Irwin, 1957)
    r     = sqrt(x² + y²)
    θ     = atan2(y, x)
    ν     = λ / (2(λ+μ))          (Poisson's ratio, plane strain)
    κ     = 3 - 4ν                 (Kolosov constant)
    u_x   = K_I/(2μ) √(r/2π) cos(θ/2) (κ - cos θ)
    u_y   = K_I/(2μ) √(r/2π) sin(θ/2) (κ - cos θ)

Domain: [0, 1]² with crack tip at the corner (0, 0).
The crack face (θ = ±π) lies entirely outside the domain (crack along
negative x-axis, outside [0,1]²), so the solution is single-valued
throughout and the only singularity is the √r behaviour near (0, 0).

Strategy: same Gauss-Seidel iteration as linear_elasticity_2D.py, with
f_body = 0 so the only right-hand side comes from the cross-derivative
coupling terms.

    Iteration k:
        s_x^(k) = -(λ+μ) D_xy u_y^(k-1)
        u_x^(k) = solve( (λ+2μ)∂_xx + μ∂_yy, bc_x, source=s_x^(k) )

        s_y^(k) = -(λ+μ) D_xy u_x^(k)
        u_y^(k) = solve( μ∂_xx + (λ+2μ)∂_yy, bc_y, source=s_y^(k) )

Expected behaviour:
    - Convergence is algebraic O(p^{-1/2}) in L^∞ (vs spectral-exponential
      for smooth problems) because the corner patch must approximate √r with
      polynomials.
    - Finer meshes (larger L) shrink the singular corner patch and lower the
      overall error floor at fixed p.

Usage:
    python examples/linear_elasticity_crack_tip_2D.py
    python examples/linear_elasticity_crack_tip_2D.py --p_vals 4 6 8 12 16 --l_vals 2 3 4
    python examples/linear_elasticity_crack_tip_2D.py --max_iter 50 --tol 1e-14
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

# ---------------------------------------------------------------------------
# Lamé constants and domain
# ---------------------------------------------------------------------------
LAMBDA = 1.0   # first Lamé parameter
MU     = 1.0   # shear modulus
K_I    = 1.0   # Mode I stress-intensity factor (normalisation)

XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0

# Derived material constants
_NU    = LAMBDA / (2 * (LAMBDA + MU))   # Poisson's ratio (plane strain)
_KAPPA = 3 - 4 * _NU                    # Kolosov constant


# ---------------------------------------------------------------------------
# Williams Mode I exact solution
# ---------------------------------------------------------------------------

def williams_ux(pts: jnp.ndarray) -> jnp.ndarray:
    """
    u_x from Mode I Williams expansion.
    pts: shape (..., 2), columns = (x, y).
    u_x = K_I/(2μ) √(r/2π) cos(θ/2) (κ - cosθ)
    """
    x, y  = pts[..., 0], pts[..., 1]
    r     = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    factor = K_I / (2 * MU) * jnp.sqrt(r / (2 * jnp.pi))
    return factor * jnp.cos(theta / 2) * (_KAPPA - jnp.cos(theta))


def williams_uy(pts: jnp.ndarray) -> jnp.ndarray:
    """
    u_y from Mode I Williams expansion.
    u_y = K_I/(2μ) √(r/2π) sin(θ/2) (κ - cosθ)
    """
    x, y  = pts[..., 0], pts[..., 1]
    r     = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)
    factor = K_I / (2 * MU) * jnp.sqrt(r / (2 * jnp.pi))
    return factor * jnp.sin(theta / 2) * (_KAPPA - jnp.cos(theta))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mode I crack-tip benchmark for 2D linear elasticity (jaxhps)."
    )
    parser.add_argument("--output_dir", type=str,
                        default="data/examples/linear_elasticity_crack_tip_2D")
    parser.add_argument("--p_vals", type=int, nargs="+",
                        default=[3, 4, 6, 8, 12, 16])
    parser.add_argument("--l_vals", type=int, nargs="+",
                        default=[2, 3, 4])
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum Gauss-Seidel iterations.")
    parser.add_argument("--tol", type=float, default=1e-14,
                        help="L-inf convergence tolerance on displacement update.")
    parser.add_argument(
        "--stress_clip_percentile",
        type=float,
        default=99.0,
        help="Upper percentile used to clip stress contours for visualization.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Gauss-Seidel solver (zero body force)
# ---------------------------------------------------------------------------

def gauss_seidel_crack_tip(
    domain: Domain,
    max_iter: int = 50,
    tol: float = 1e-14,
) -> tuple[jnp.ndarray, jnp.ndarray, list[float], float, float]:
    """
    Solve 2D linear elasticity (no body force) on domain via GS iteration.

    Returns:
        u_x, u_y    : solution arrays of shape (n_leaves, p^2)
        residuals   : list of L-inf update norms per iteration
        t_build     : wall time (s) for the two build_solver calls
        t_solve     : wall time (s) for all GS iterations
    """
    ones = jnp.ones_like(domain.interior_points[..., 0])

    # Exact boundary conditions from Williams expansion
    bc_ux = jnp.expand_dims(williams_ux(domain.boundary_points), axis=-1)
    bc_uy = jnp.expand_dims(williams_uy(domain.boundary_points), axis=-1)

    # Build both operators once with source=None
    t0_build = time.perf_counter()

    pde_ux = PDEProblem(
        domain=domain,
        D_xx_coefficients=(LAMBDA + 2 * MU) * ones,
        D_yy_coefficients=MU * ones,
    )
    build_solver(pde_ux)

    pde_uy = PDEProblem(
        domain=domain,
        D_xx_coefficients=MU * ones,
        D_yy_coefficients=(LAMBDA + 2 * MU) * ones,
    )
    build_solver(pde_uy)

    t_build = time.perf_counter() - t0_build

    D_xy = pde_ux.D_xy   # mixed-derivative operator on each leaf

    # Initialise: u^(0) = 0
    u_x = jnp.zeros_like(ones)
    u_y = jnp.zeros_like(ones)

    residuals = []
    t0_solve = time.perf_counter()

    for k in range(max_iter):
        u_x_old = u_x
        u_y_old = u_y

        # Source for u_x: -(λ+μ) ∂_xy u_y
        u_y_xy = jnp.einsum("ij,lj->li", D_xy, u_y)
        s_x    = -(LAMBDA + MU) * u_y_xy
        u_x    = solve(pde_ux, bc_ux, source=s_x)[..., 0]

        # Source for u_y: -(λ+μ) ∂_xy u_x  (updated u_x)
        u_x_xy = jnp.einsum("ij,lj->li", D_xy, u_x)
        s_y    = -(LAMBDA + MU) * u_x_xy
        u_y    = solve(pde_uy, bc_uy, source=s_y)[..., 0]

        res = float(jnp.max(jnp.abs(u_x - u_x_old) + jnp.abs(u_y - u_y_old)))
        residuals.append(res)
        logging.debug("GS iter %i: update = %.4e", k + 1, res)

        if res < tol:
            logging.info("GS converged in %i iterations (update=%.4e)", k + 1, res)
            break
    else:
        logging.warning("GS did not converge in %i iterations (last update=%.4e)",
                        max_iter, res)

    t_solve = time.perf_counter() - t0_solve
    return u_x, u_y, residuals, t_build, t_solve


# ---------------------------------------------------------------------------
# hp-convergence sweep
# ---------------------------------------------------------------------------

def run_convergence(
    l_vals: list,
    p_vals: list,
    max_iter: int,
    tol: float,
) -> dict:
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

            root   = DiscretizationNode2D(xmin=XMIN, xmax=XMAX, ymin=YMIN, ymax=YMAX)
            domain = Domain(p=p, q=max(1, p - 2), root=root, L=l)

            u_x, u_y, residuals, tb, ts = gauss_seidel_crack_tip(
                domain, max_iter=max_iter, tol=tol
            )

            exact_ux = williams_ux(domain.interior_points)
            exact_uy = williams_uy(domain.interior_points)

            # Absolute L-inf error (exact solution is zero at tip, so relative
            # norm uses the max of the exact field as the scale).
            abs_err_ux = float(jnp.max(jnp.abs(u_x - exact_ux)))
            abs_err_uy = float(jnp.max(jnp.abs(u_y - exact_uy)))
            scale_ux   = float(jnp.max(jnp.abs(exact_ux)))
            scale_uy   = float(jnp.max(jnp.abs(exact_uy)))

            err_ux = abs_err_ux / scale_ux
            err_uy = abs_err_uy / scale_uy

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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v"]
COLORS  = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_error_vs_p(
    l_vals: list,
    p_vals: list,
    errors_ux: np.ndarray,
    errors_uy: np.ndarray,
    output_dir: str,
) -> None:
    """
    Log-log plot of relative L-inf error vs p, one line per L.
    Algebraic convergence shows as a straight line on this scale.
    Reference lines for O(p^{-0.5}) and O(p^{-1}) are overlaid.
    """
    pv = np.array(p_vals, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, errors in zip(axes, [errors_ux, errors_uy]):
        for i, l in enumerate(l_vals):
            ax.loglog(
                pv, errors[i],
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=f"$L={l}$",
            )

        ax.set_xlabel("Polynomial degree $p$")
        ax.set_ylabel("Relative $L^\\infty$ error")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle=":")

    fig.tight_layout()
    fp = os.path.join(output_dir, "error_vs_p.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Error vs p plot saved to: %s", fp)


def plot_runtimes(
    l_vals: list,
    p_vals: list,
    t_build: np.ndarray,
    t_solve: np.ndarray,
    output_dir: str,
) -> None:
    t_total = t_build + t_solve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, l in enumerate(l_vals):
        ax.plot(p_vals, t_total[i],
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=f"$L={l}$")
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel("Total wall time (s)")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fp = os.path.join(output_dir, "runtimes.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Runtime plot saved to: %s", fp)


def plot_iterations(
    l_vals: list,
    p_vals: list,
    n_iters: np.ndarray,
    output_dir: str,
) -> None:
    """Line plot of Gauss-Seidel iteration count vs p, one line per L."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, l in enumerate(l_vals):
        ax.plot(
            p_vals,
            n_iters[i],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=f"$L={l}$",
        )
    ax.set_xlabel("Polynomial degree $p$")
    ax.set_ylabel("Iterations")
    ax.legend()
    ax.grid(True, linestyle=":")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fp = os.path.join(output_dir, "iterations.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Iteration plot saved to: %s", fp)


def plot_solution_contours(
    domain: Domain,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    output_dir: str,
    n_levels: int = 25,
) -> None:
    """Filled contours for numerical u_x and u_y on Chebyshev interior points."""
    pts = np.array(domain.interior_points).reshape(-1, 2)
    x_pts, y_pts = pts[:, 0], pts[:, 1]

    ux_vals  = np.array(u_x).ravel()
    uy_vals  = np.array(u_y).ravel()
    triang = mtri.Triangulation(x_pts, y_pts)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    datasets = [
        (ux_vals, r"$u_x$", "RdBu_r"),
        (uy_vals, r"$u_y$", "RdBu_r"),
    ]

    for ax, (vals, title, cmap) in zip(np.ravel(axes), datasets):
        tcf = ax.tricontourf(triang, vals, levels=n_levels, cmap=cmap)
        plt.colorbar(tcf, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        # Mark the crack tip
        ax.plot(0, 0, "r*", markersize=10)

    fig.tight_layout()
    fp = os.path.join(output_dir, "solution_contours.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Solution contour plot saved to: %s", fp)


def plot_displacement_magnitude(
    domain: Domain,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    output_dir: str,
    n_levels: int = 25,
) -> None:
    """
    Single panel: displacement magnitude |u|, showing the √r gradient from
    the crack tip.  Annotated with polar angle isolines.
    """
    pts    = np.array(domain.interior_points).reshape(-1, 2)
    x_pts, y_pts = pts[:, 0], pts[:, 1]
    mag    = np.sqrt(np.array(u_x).ravel()**2 + np.array(u_y).ravel()**2)
    triang = mtri.Triangulation(x_pts, y_pts)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    tcf = ax.tricontourf(triang, mag, levels=n_levels, cmap="plasma")
    plt.colorbar(tcf, ax=ax, label=r"$|u|$")
    ax.set_title(r"$|\mathbf{u}|$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.plot(0, 0, "w*", markersize=12)
    fig.tight_layout()
    fp = os.path.join(output_dir, "displacement_magnitude.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Displacement magnitude plot saved to: %s", fp)


def plot_stress_contours(
    domain: Domain,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    output_dir: str,
    n_levels: int = 25,
    clip_percentile: float = 99.0,
) -> None:
    """Filled stress contours with robust clipping to handle crack-tip blow-up."""
    ones = jnp.ones_like(domain.interior_points[..., 0])
    pde = PDEProblem(
        domain=domain,
        D_xx_coefficients=ones,
        D_yy_coefficients=ones,
    )

    u_x_x = jnp.einsum("ij,lj->li", pde.D_x, u_x)
    u_x_y = jnp.einsum("ij,lj->li", pde.D_y, u_x)
    u_y_x = jnp.einsum("ij,lj->li", pde.D_x, u_y)
    u_y_y = jnp.einsum("ij,lj->li", pde.D_y, u_y)

    eps_xx = u_x_x
    eps_yy = u_y_y
    eps_xy = 0.5 * (u_x_y + u_y_x)
    tr_eps = eps_xx + eps_yy

    sigma_xx = LAMBDA * tr_eps + 2 * MU * eps_xx
    sigma_yy = LAMBDA * tr_eps + 2 * MU * eps_yy
    sigma_xy = 2 * MU * eps_xy
    sigma_vm = jnp.sqrt(
        jnp.maximum(sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * sigma_xy**2, 0.0)
    )

    pts = np.array(domain.interior_points).reshape(-1, 2)
    x_pts, y_pts = pts[:, 0], pts[:, 1]
    triang = mtri.Triangulation(x_pts, y_pts)

    sigma_xx_vals = np.array(sigma_xx).ravel()
    sigma_yy_vals = np.array(sigma_yy).ravel()
    sigma_xy_vals = np.array(sigma_xy).ravel()
    sigma_vm_vals = np.array(sigma_vm).ravel()

    signed_fields = [sigma_xx_vals, sigma_yy_vals, sigma_xy_vals]
    signed_limits = []
    for vals in signed_fields:
        vmax = np.percentile(np.abs(vals), clip_percentile)
        signed_limits.append((-vmax, vmax))

    vm_low, vm_high = np.percentile(sigma_vm_vals, [5.0, clip_percentile])

    datasets = [
        (sigma_xx_vals, signed_limits[0], "coolwarm", r"$\sigma_{xx}$"),
        (sigma_yy_vals, signed_limits[1], "coolwarm", r"$\sigma_{yy}$"),
        (sigma_xy_vals, signed_limits[2], "coolwarm", r"$\sigma_{xy}$"),
        (sigma_vm_vals, (vm_low, vm_high), "magma", r"$\sigma_{\mathrm{vm}}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (vals, lims, cmap, title) in zip(axes.ravel(), datasets):
        vmin, vmax = lims
        tcf = ax.tricontourf(
            triang,
            np.clip(vals, vmin, vmax),
            levels=n_levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(tcf, ax=ax, shrink=0.85)
        ax.set_title(title)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.plot(0, 0, "c*", markersize=10)

    fig.tight_layout()
    fp = os.path.join(output_dir, "stress_contours.png")
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    logging.info("Stress contour plot saved to: %s", fp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = setup_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s:jaxhps: %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Output directory: %s", args.output_dir)
    logging.info("Lamé constants: lambda=%.2f, mu=%.2f", LAMBDA, MU)
    logging.info("Kolosov constant kappa=%.4f (nu=%.4f)", _KAPPA, _NU)
    logging.info("GS params: max_iter=%i, tol=%.2e", args.max_iter, args.tol)

    results = run_convergence(
        l_vals=args.l_vals,
        p_vals=args.p_vals,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    # Save results
    mat_path = os.path.join(args.output_dir, "results.mat")
    savemat(mat_path, {
        "errors_ux":   results["errors_ux"],
        "errors_uy":   results["errors_uy"],
        "t_build":     results["t_build"],
        "t_solve":     results["t_solve"],
        "n_iters":     results["n_iters"],
        "l_vals":      np.array(args.l_vals),
        "p_vals":      np.array(args.p_vals),
        "lame_lambda": LAMBDA,
        "lame_mu":     MU,
        "K_I":         K_I,
        "kappa":       _KAPPA,
        "nu":          _NU,
    })
    logging.info("Results saved to: %s", mat_path)

    # Plots
    plot_error_vs_p(args.l_vals, args.p_vals,
                    results["errors_ux"], results["errors_uy"],
                    args.output_dir)

    plot_runtimes(args.l_vals, args.p_vals,
                  results["t_build"], results["t_solve"],
                  args.output_dir)

    plot_iterations(args.l_vals, args.p_vals,
                    results["n_iters"],
                    args.output_dir)

    plot_solution_contours(
        results["last_domain"],
        results["last_ux"],
        results["last_uy"],
        args.output_dir,
    )

    plot_displacement_magnitude(
        results["last_domain"],
        results["last_ux"],
        results["last_uy"],
        args.output_dir,
    )

    plot_stress_contours(
        results["last_domain"],
        results["last_ux"],
        results["last_uy"],
        args.output_dir,
        clip_percentile=args.stress_clip_percentile,
    )


if __name__ == "__main__":
    main()
