#!/usr/bin/env python3
"""Simple example to demonstrate jaxhps functionality."""

import jax.numpy as jnp
import jaxhps

print("=" * 60)
print("jaxhps Simple Example - Solving a Poisson PDE")
print("=" * 60)

print("\n1. Creating domain...")
root = jaxhps.DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

domain = jaxhps.Domain(
    p=16,  # polynomial degree
    q=14,  # quadrature degree
    root=root,
    L=3,   # number of refinement levels
)
print(f"   ✓ Domain created with {len(domain.interior_points)} interior points")

print("\n2. Setting up PDE...")
source_term = jnp.zeros_like(domain.interior_points[..., 0])
D_xx_coeffs = jnp.ones_like(domain.interior_points[..., 0])
D_yy_coeffs = jnp.ones_like(domain.interior_points[..., 0])

pde_problem = jaxhps.PDEProblem(
    domain=domain,
    source=source_term,
    D_xx_coefficients=D_xx_coeffs,
    D_yy_coefficients=D_yy_coeffs,
)
print("   ✓ PDE problem created: -Δu = 0 (Laplace equation)")

print("\n3. Building solver...")
jaxhps.build_solver(pde_problem=pde_problem)
print("   ✓ Solver built successfully!")

print("\n4. Setting boundary conditions...")
boundary_data = (
    domain.boundary_points[..., 0] ** 2 - domain.boundary_points[..., 1] ** 2
)
print(f"   ✓ Boundary data set (x² - y²)")

print("\n5. Solving PDE...")
solution = jaxhps.solve(pde_problem=pde_problem, boundary_data=boundary_data)
print(f"   ✓ Solution computed!")
print(f"   Solution range: [{jnp.min(solution):.6e}, {jnp.max(solution):.6e}]")
print(f"   Solution L∞ norm: {jnp.max(jnp.abs(solution)):.6e}")

print("\n" + "=" * 60)
print("✓ Example completed successfully!")
print("=" * 60)
