#!/usr/bin/env python3

"""
Lid-Driven Cavity Flow

Solves the steady-state incompressible Navier-Stokes equations:
    Continuity: ∇·u = 0
    Momentum: (u·∇)u = -∇p + (1/Re)∇²u

Where:
    u = x-velocity (horizontal velocity component)
    v = y-velocity (vertical velocity component)
    p = pressure

Boundary conditions:
    Top wall (y=1): u = 1, v = 0  (moving lid)
    Other walls: Free-slip (normal velocity = 0, zero normal gradient for tangential velocity)
    Pressure: Reference point at bottom-left corner
"""

import argparse
import csv
import os

import numpy as np

import odil
from odil import plotutil, printlog

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def operator(ctx):
    """
    Defines the Navier-Stokes equations as residuals.
    
    The operator function is the core of ODIL - it defines your PDE
    as discrete residuals that should equal zero.
    """
    domain = ctx.domain
    extra = ctx.extra
    args = extra.args
    mod = ctx.mod
    
    # Get grid spacing and indices
    dx, dy = ctx.step()
    ix, iy = ctx.indices()
    nx, ny = ctx.size()
    
    # Get fields (cell-centered)
    u = ctx.field("u")  # x-velocity
    v = ctx.field("v")  # y-velocity
    p = ctx.field("p")  # pressure
    
    # Get neighboring values using shifts
    # Format: ctx.field("name", shift_x, shift_y)
    u_w = ctx.field("u", -1, 0)  # west (left)
    u_e = ctx.field("u", 1, 0)   # east (right)
    u_s = ctx.field("u", 0, -1)  # south (bottom)
    u_n = ctx.field("u", 0, 1)   # north (top)
    
    v_w = ctx.field("v", -1, 0)
    v_e = ctx.field("v", 1, 0)
    v_s = ctx.field("v", 0, -1)
    v_n = ctx.field("v", 0, 1)
    
    p_w = ctx.field("p", -1, 0)
    p_e = ctx.field("p", 1, 0)
    p_s = ctx.field("p", 0, -1)
    p_n = ctx.field("p", 0, 1)
    
    # Apply boundary conditions to stencil values
    zero = ctx.cast(0)
    one = ctx.cast(1)
    
    # FREE-SLIP BOUNDARY CONDITIONS:
    # Normal velocity = 0, tangential velocity has zero normal gradient (mirror condition)
    
    # Top wall (iy == ny-1): u = 1 (moving lid), v = 0 (normal = 0)
    u_n = mod.where(iy == ny - 1, one, u_n)
    v_n = mod.where(iy == ny - 1, zero, v_n)
    
    # Bottom wall (iy == 0): v = 0 (normal = 0), u has zero gradient in y (free slip)
    # Mirror condition: u_s = u (tangential velocity equals interior value)
    v_s = mod.where(iy == 0, zero, v_s)
    u_s = mod.where(iy == 0, u, u_s)  # Free slip: mirror u
    
    # Left wall (ix == 0): u = 0 (normal = 0), v has zero gradient in x (free slip)
    # Mirror condition: v_w = v (tangential velocity equals interior value)
    u_w = mod.where(ix == 0, zero, u_w)
    v_w = mod.where(ix == 0, v, v_w)  # Free slip: mirror v
    
    # Right wall (ix == nx-1): u = 0 (normal = 0), v has zero gradient in x (free slip)
    # Mirror condition: v_e = v (tangential velocity equals interior value)
    u_e = mod.where(ix == nx - 1, zero, u_e)
    v_e = mod.where(ix == nx - 1, v, v_e)  # Free slip: mirror v
    
    # Compute derivatives using finite differences
    # Laplacian (diffusion term)
    u_xx = (u_e - 2 * u + u_w) / (dx ** 2)
    u_yy = (u_n - 2 * u + u_s) / (dy ** 2)
    lap_u = u_xx + u_yy
    
    v_xx = (v_e - 2 * v + v_w) / (dx ** 2)
    v_yy = (v_n - 2 * v + v_s) / (dy ** 2)
    lap_v = v_xx + v_yy
    
    # Pressure gradients (central difference)
    p_x = (p_e - p_w) / (2 * dx)
    p_y = (p_n - p_s) / (2 * dy)
    
    # Advection terms using upwind scheme
    # For u-equation: u*u_x + v*u_y
    u_x = mod.where(
        u > 0,
        (u - u_w) / dx,  # backward difference
        mod.where(
            u < 0,
            (u_e - u) / dx,  # forward difference
            (u_e - u_w) / (2 * dx)  # central difference
        )
    )
    u_y = mod.where(
        v > 0,
        (u - u_s) / dy,
        mod.where(
            v < 0,
            (u_n - u) / dy,
            (u_n - u_s) / (2 * dy)
        )
    )
    adv_u = u * u_x + v * u_y
    
    # For v-equation: u*v_x + v*v_y
    v_x = mod.where(
        u > 0,
        (v - v_w) / dx,
        mod.where(
            u < 0,
            (v_e - v) / dx,
            (v_e - v_w) / (2 * dx)
        )
    )
    v_y = mod.where(
        v > 0,
        (v - v_s) / dy,
        mod.where(
            v < 0,
            (v_n - v) / dy,
            (v_n - v_s) / (2 * dy)
        )
    )
    adv_v = u * v_x + v * v_y
    
    # Reynolds number
    Re = args.Re
    
    # Momentum equations: (u·∇)u = -∇p + (1/Re)∇²u
    # Steady state: (u·∇)u + ∇p - (1/Re)∇²u = 0
    fu = adv_u + p_x - (1.0 / Re) * lap_u
    fv = adv_v + p_y - (1.0 / Re) * lap_v
    
    # Continuity equation: ∇·u = 0
    # Use central differences for divergence
    u_x_div = (u_e - u_w) / (2 * dx)
    v_y_div = (v_n - v_s) / (2 * dy)
    fdiv = u_x_div + v_y_div
    
    # Enforce boundary conditions in residuals
    # Top wall: u = 1, v = 0
    fu = mod.where(iy == ny - 1, (u - one) / dx, fu)
    fv = mod.where(iy == ny - 1, v / dx, fv)
    
    # Bottom wall: v = 0 (normal), ∂u/∂y = 0 (free slip)
    # Enforce v = 0 and zero gradient for u (already handled by mirror condition in stencil)
    fv = mod.where(iy == 0, v / dx, fv)
    # u has zero gradient, so no explicit constraint needed (mirror handles it)
    
    # Left wall: u = 0 (normal), ∂v/∂x = 0 (free slip)
    # Enforce u = 0 and zero gradient for v (already handled by mirror condition in stencil)
    fu = mod.where(ix == 0, u / dx, fu)
    # v has zero gradient, so no explicit constraint needed (mirror handles it)
    
    # Right wall: u = 0 (normal), ∂v/∂x = 0 (free slip)
    # Enforce u = 0 and zero gradient for v (already handled by mirror condition in stencil)
    fu = mod.where(ix == nx - 1, u / dx, fu)
    # v has zero gradient, so no explicit constraint needed (mirror handles it)
    
    # Pressure: set reference point (bottom-left corner)
    fpress = mod.where((ix == 0) & (iy == 0), p, zero)
    
    # Return residuals as list of (name, value) tuples
    res = [("fu", fu), ("fv", fv), ("fdiv", fdiv), ("fpress", fpress * args.kpress)]
    
    return res


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Lid-driven cavity flow solver"
    )
    parser.add_argument("--Nx", type=int, default=32, help="Grid size in x")
    parser.add_argument("--Ny", type=int, default=32, help="Grid size in y")
    parser.add_argument("--Re", type=float, default=100.0, help="Reynolds number")
    parser.add_argument("--kpress", type=float, default=1.0, help="Pressure reference weight")
    parser.add_argument("--plot", type=int, default=1, help="Enable plotting")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    
    parser.set_defaults(outdir="out_lid_cavity")
    parser.set_defaults(echo=1)
    parser.set_defaults(frames=10, plot_every=100, report_every=50, history_every=10)
    parser.set_defaults(optimizer="adam", lr=1e-3)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(nlvl=None)  # Auto-detect number of levels
    parser.set_defaults(double=1)
    
    return parser.parse_args()


def plot_func(problem, state, epoch, frame, cbinfo=None):
    """Plot velocity field and streamlines"""
    if not HAS_MATPLOTLIB:
        printlog("matplotlib not available, skipping plot")
        return
    
    domain = problem.domain
    args = problem.extra.args
    
    # Get fields
    u = domain.field(state, "u")
    v = domain.field(state, "v")
    p = domain.field(state, "p")
    
    # Get coordinates
    # Note: domain.points returns arrays with indexing="ij" (first index is x, second is y)
    x, y = domain.points("x", "y", loc="cc")
    # For plotting, we need to ensure x and y are properly shaped
    # x and y are already in the correct format from ODIL
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Velocity magnitude
    ax = axes[0]
    vel_mag = np.sqrt(u**2 + v**2)
    im = ax.contourf(x, y, vel_mag, levels=20, cmap="viridis")
    ax.set_title(f"Velocity Magnitude (epoch {epoch})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Streamlines (following standard matplotlib pattern)
    ax = axes[1]
    # Convert to numpy arrays
    x_np = np.array(x)
    y_np = np.array(y)
    u_np = np.array(u)
    v_np = np.array(v)
    
    # Extract 1D coordinate arrays
    # In ODIL with 'ij' indexing: x varies along first dim, y along second dim
    x_1d = x_np[:, 0]  # x coordinates (same for all rows, take first column)
    y_1d = y_np[0, :]  # y coordinates (same for all columns, take first row)
    
    # Create meshgrid with 'xy' indexing (default, standard for matplotlib)
    X, Y = np.meshgrid(x_1d, y_1d)
    
    # Velocity arrays need to be transposed to match 'xy' indexing
    # ODIL: u[i, j] where i is x-index, j is y-index
    # streamplot: U[i, j] where i is y-index, j is x-index
    U = u_np.T  # Transpose to match streamplot format
    V = v_np.T  # Transpose to match streamplot format
    
    # Compute velocity magnitude for coloring
    speed = np.sqrt(U**2 + V**2)
    
    # Create streamlines colored by velocity magnitude
    strm = ax.streamplot(
        X, Y, U, V,
        density=1.5,
        color=speed,
        cmap='viridis',
        linewidth=1.5,
    )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamlines')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(strm.lines, ax=ax, label='Velocity Magnitude')
    
    # Plot 3: Pressure
    ax = axes[2]
    im = ax.contourf(x, y, p, levels=20, cmap="RdBu_r")
    ax.set_title("Pressure")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plotutil.savefig(fig, f"field_{frame:05d}", printf=printlog)
    plt.close(fig)
    
    # Plot centerlines
    plot_centerlines(domain, state, epoch, frame, args)


def plot_centerlines(domain, state, epoch, frame, args):
    """Plot u and v velocity profiles along centerlines"""
    if not HAS_MATPLOTLIB:
        return
    
    # Get fields
    u = domain.field(state, "u")
    v = domain.field(state, "v")
    
    # Get coordinates
    x, y = domain.points("x", "y", loc="cc")
    
    # Convert to numpy arrays if needed (for JAX compatibility)
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    
    # Find center indices (closest to x=0.5 and y=0.5)
    # In ODIL with indexing="ij": first dimension is x, second is y
    # So x varies along first dimension (rows), y varies along second dimension (columns)
    nx, ny = domain.cshape
    
    # For vertical centerline (x = 0.5): find row index where x is closest to 0.5
    # x varies along rows (first dimension), so check any column
    x_col = x[:, 0]  # x coordinates along first column (x varies here)
    ix_center = np.argmin(np.abs(x_col - 0.5))
    
    # For horizontal centerline (y = 0.5): find column index where y is closest to 0.5
    # y varies along columns (second dimension), so check any row
    y_row = y[0, :]  # y coordinates along first row (y varies here)
    iy_center = np.argmin(np.abs(y_row - 0.5))
    
    # Extract centerline data
    # u (x-velocity) along vertical centerline (x = 0.5)
    # Extract the entire row at ix_center (all y positions at fixed x)
    u_centerline = u[ix_center, :]  # All columns (y positions), fixed x row
    y_centerline = y[ix_center, :]  # y coordinates along that row
    
    # v (y-velocity) along horizontal centerline (y = 0.5)
    # Extract the entire column at iy_center (all x positions at fixed y)
    v_centerline = v[:, iy_center]  # All rows (x positions), fixed y column
    x_centerline = x[:, iy_center]  # x coordinates along that column
    
    # Save to CSV - write both centerlines
    csv_path = f"centerlines_{frame:05d}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["vertical_centerline_x=0.5", "", "horizontal_centerline_y=0.5", ""])
        writer.writerow(["y", "u_x_velocity", "x", "v_y_velocity"])
        # Write data - both centerlines have same length (nx = ny for square domain)
        max_len = max(len(y_centerline), len(x_centerline))
        for i in range(max_len):
            y_val = y_centerline[i] if i < len(y_centerline) else ""
            u_val = u_centerline[i] if i < len(u_centerline) else ""
            x_val = x_centerline[i] if i < len(x_centerline) else ""
            v_val = v_centerline[i] if i < len(v_centerline) else ""
            writer.writerow([y_val, u_val, x_val, v_val])
    printlog(f"Saved centerline data to {csv_path}")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot u (x-velocity) along vertical centerline (x = 0.5)
    ax = axes[0]
    ax.plot(u_centerline, y_centerline, "b-", linewidth=2, label=f"u (x-velocity) at x=0.5, epoch {epoch}")
    ax.set_xlabel("u (x-velocity)")
    ax.set_ylabel("y")
    ax.set_title("x-velocity (u) along vertical centerline (x=0.5)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Set appropriate limits based on data
    u_min, u_max = np.min(u_centerline), np.max(u_centerline)
    u_range = u_max - u_min
    ax.set_xlim(u_min - 0.1 * u_range if u_range > 0 else -0.1, u_max + 0.1 * u_range if u_range > 0 else 1.1)
    ax.set_ylim(0, 1)  # y goes from 0 to 1
    
    # Plot v (y-velocity) along horizontal centerline (y = 0.5)
    ax = axes[1]
    ax.plot(x_centerline, v_centerline, "r-", linewidth=2, label=f"v (y-velocity) at y=0.5, epoch {epoch}")
    ax.set_xlabel("x")
    ax.set_ylabel("v (y-velocity)")
    ax.set_title("y-velocity (v) along horizontal centerline (y=0.5)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Set appropriate limits based on data
    v_min, v_max = np.min(v_centerline), np.max(v_centerline)
    v_range = v_max - v_min
    ax.set_xlim(0, 1)  # x goes from 0 to 1
    ax.set_ylim(v_min - 0.1 * v_range if v_range > 0 else -0.1, v_max + 0.1 * v_range if v_range > 0 else 0.1)
    
    plt.tight_layout()
    plotutil.savefig(fig, f"centerlines_{frame:05d}", printf=printlog)
    plt.close(fig)


def plot_error(problem, state, epoch, history, cbinfo):
    """Add error metrics to history and create error plots"""
    if history is None:
        return
    
    domain = problem.domain
    args = problem.extra.args
    
    # Get fields
    u = domain.field(state, "u")
    v = domain.field(state, "v")
    
    # Compute residual norms (already in pinfo, but we can add custom metrics)
    # For lid-driven cavity, we can track:
    # 1. Maximum velocity (should be ~1 at top wall)
    # 2. Divergence error
    # 3. Residual norms
    
    # Maximum velocity
    vel_mag = np.sqrt(u**2 + v**2)
    max_vel = float(np.array(np.max(vel_mag)))  # Convert to Python float
    history.append("max_velocity", max_vel)
    
    # Centerline values for monitoring
    # Get coordinates to find actual center
    x, y_coords = domain.points("x", "y", loc="cc")
    nx, ny = domain.cshape
    
    # Find center indices
    x_mid = x[ny // 2, :]
    y_mid = y_coords[:, nx // 2]
    ix_center = np.argmin(np.abs(x_mid - 0.5))
    iy_center = np.argmin(np.abs(y_mid - 0.5))
    
    u_center = u[iy_center, ix_center]  # u at center
    v_center = v[iy_center, ix_center]  # v at center
    # Convert to numpy scalar for history
    u_center = float(np.array(u_center))
    v_center = float(np.array(v_center))
    history.append("u_center", u_center)
    history.append("v_center", v_center)


def plot_convergence(cbinfo):
    """Plot convergence history from CSV"""
    if not HAS_MATPLOTLIB:
        return
    
    csv_path = "train.csv"
    if not os.path.exists(csv_path):
        printlog(f"CSV file {csv_path} not found, skipping convergence plot")
        return
    
    try:
        # Load CSV data
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        
        if len(data) == 0:
            printlog("CSV file is empty, skipping convergence plot")
            return
        
        # Handle both scalar and array cases
        if data.ndim == 0:
            epochs = np.array([data["epoch"]])
        else:
            epochs = data["epoch"]
        
        # Create convergence plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Residual norms
        ax = axes[0, 0]
        if "norm_fu" in data.dtype.names:
            ax.semilogy(epochs, data["norm_fu"], "b-", label="fu (u-momentum)", linewidth=2)
        if "norm_fv" in data.dtype.names:
            ax.semilogy(epochs, data["norm_fv"], "r-", label="fv (v-momentum)", linewidth=2)
        if "norm_fdiv" in data.dtype.names:
            ax.semilogy(epochs, data["norm_fdiv"], "g-", label="fdiv (continuity)", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Residual Norm")
        ax.set_title("Residual Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Loss
        ax = axes[0, 1]
        if "loss" in data.dtype.names:
            ax.semilogy(epochs, data["loss"], "k-", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Convergence")
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Maximum velocity
        ax = axes[1, 0]
        if "max_velocity" in data.dtype.names:
            ax.plot(epochs, data["max_velocity"], "b-", linewidth=2)
            ax.axhline(y=1.0, color="r", linestyle="--", label="Expected (top wall)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Max Velocity")
            ax.set_title("Maximum Velocity")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Center velocities
        ax = axes[1, 1]
        if "u_center" in data.dtype.names:
            ax.plot(epochs, data["u_center"], "b-", label="u center", linewidth=2)
        if "v_center" in data.dtype.names:
            ax.plot(epochs, data["v_center"], "r-", label="v center", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Velocity")
        ax.set_title("Center Point Velocities")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plotutil.savefig(fig, "convergence", printf=printlog)
        plt.close(fig)
        
    except Exception as e:
        printlog(f"Error plotting convergence: {e}")


def make_problem(args):
    """
    Create the problem domain and initial state.
    
    This function sets up:
    1. The computational domain (grid)
    2. The state variables (fields to solve for)
    3. Any extra data needed by the operator
    """
    dtype = np.float64 if args.double else np.float32
    
    # Create domain: square cavity [0,1] x [0,1]
    domain = odil.Domain(
        cshape=(args.Nx, args.Ny),
        dimnames=["x", "y"],
        lower=(0, 0),
        upper=(1, 1),
        dtype=dtype,
        multigrid=args.multigrid,
        mg_interp=args.mg_interp,
        mg_axes=[True, True],
        mg_nlvl=args.nlvl,
    )
    
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)
    
    # Initialize state with zero velocity and pressure
    from odil import Field
    
    state = odil.State(
        fields={
            "u": Field(np.zeros(domain.size(loc="cc")), loc="cc"),  # x-velocity
            "v": Field(np.zeros(domain.size(loc="cc")), loc="cc"),  # y-velocity
            "p": Field(np.zeros(domain.size(loc="cc")), loc="cc"),  # pressure
        }
    )
    state = domain.init_state(state)  # IMPORTANT: Always call this!
    
    # Store args in extra for use in operator
    extra = argparse.Namespace()
    extra.args = args
    
    problem = odil.Problem(operator, domain, extra)
    return problem, state


def main():
    """
    Main function: sets up and runs the optimization.
    """
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    callback = odil.make_callback(
        problem,
        args,
        plot_func=plot_func if args.plot else None,
        history_func=plot_error,
    )
    odil.util.optimize(args, args.optimizer, problem, state, callback)
    
    # Plot final convergence
    if args.plot and HAS_MATPLOTLIB:
        plot_convergence(callback.cbinfo)


if __name__ == "__main__":
    main()

