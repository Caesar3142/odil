# How to Use the ODIL Template

This guide explains how to solve a PDE problem using the ODIL framework template, using the lid-driven cavity as an example.

## Step-by-Step Guide

### Step 1: Define Your Operator Function

The **operator** is the core of your ODIL implementation. It defines your PDE as discrete residuals.

```python
def operator(ctx):
    """
    ctx provides access to:
    - ctx.domain: The computational domain
    - ctx.field("name"): Get field values
    - ctx.field("name", shift_x, shift_y): Get field at shifted location
    - ctx.points("x", "y"): Get coordinates
    - ctx.indices("x", "y"): Get indices (for boundary conditions)
    - ctx.step(): Get grid spacing (dx, dy, ...)
    - ctx.size(): Get grid size (nx, ny, ...)
    - ctx.mod: Math module (use mod.where, mod.cast, etc.)
    - ctx.extra: Extra data you passed to Problem
    """
    mod = ctx.mod
    dx, dy = ctx.step()
    ix, iy = ctx.indices()
    nx, ny = ctx.size()
    
    # 1. Get your fields
    u = ctx.field("u")
    v = ctx.field("v")
    
    # 2. Get neighboring values (for finite differences)
    u_w = ctx.field("u", -1, 0)  # west (left)
    u_e = ctx.field("u", 1, 0)   # east (right)
    u_s = ctx.field("u", 0, -1)   # south (bottom)
    u_n = ctx.field("u", 0, 1)    # north (top)
    
    # 3. Apply boundary conditions to stencil values
    zero = mod.cast(0)
    one = mod.cast(1)
    u_n = mod.where(iy == ny - 1, one, u_n)  # Top: u = 1
    u_s = mod.where(iy == 0, zero, u_s)      # Bottom: u = 0
    
    # 4. Compute derivatives (finite differences)
    u_xx = (u_e - 2*u + u_w) / (dx**2)  # Second derivative
    u_yy = (u_n - 2*u + u_s) / (dy**2)
    lap_u = u_xx + u_yy  # Laplacian
    
    # 5. Define your PDE as a residual (should equal zero)
    fu = lap_u - source_term
    
    # 6. Enforce boundary conditions in residuals
    fu = mod.where(iy == ny - 1, (u - one) / dx, fu)
    
    # 7. Return list of (name, residual) tuples
    return [("fu", fu)]
```

### Step 2: Create the Domain

The domain defines your computational grid.

```python
domain = odil.Domain(
    cshape=(Nx, Ny),        # Grid size in cells
    dimnames=["x", "y"],    # Dimension names
    lower=(0, 0),          # Lower bounds
    upper=(1, 1),          # Upper bounds
    dtype=np.float64,       # Data type
    multigrid=True,         # Enable multigrid (optional)
)
```

### Step 3: Initialize State

State contains all the fields you want to solve for.

```python
from odil import Field

state = odil.State(
    fields={
        "u": Field(np.zeros(domain.size(loc="cc")), loc="cc"),
        "v": Field(np.zeros(domain.size(loc="cc")), loc="cc"),
    }
)
state = domain.init_state(state)  # IMPORTANT: Always call this!
```

**Field locations (`loc`):**
- `"cc"` = cell-centered in both x and y
- `"nc"` = node in x, cell in y (x-face)
- `"cn"` = cell in x, node in y (y-face)
- `"nn"` = node in both x and y

### Step 4: Create the Problem

Combine operator, domain, and extra data.

```python
extra = argparse.Namespace()
extra.args = args
extra.source = source_field  # Any extra data you need

problem = odil.Problem(operator, domain, extra)
```

### Step 5: Set Up Optimization

Create a callback for monitoring/plotting and run optimization.

```python
def plot_func(problem, state, epoch, frame, cbinfo=None):
    # Your plotting code here
    pass

callback = odil.make_callback(
    problem, 
    args, 
    plot_func=plot_func,
)

odil.util.optimize(args, "adam", problem, state, callback)
```

## Complete Template Structure

```python
#!/usr/bin/env python3
import argparse
import numpy as np
import odil

# 1. OPERATOR: Define your PDE
def operator(ctx):
    mod = ctx.mod
    dx, dy = ctx.step()
    ix, iy = ctx.indices()
    nx, ny = ctx.size()
    
    # Get fields and neighbors
    u = ctx.field("u")
    u_w = ctx.field("u", -1, 0)
    u_e = ctx.field("u", 1, 0)
    u_s = ctx.field("u", 0, -1)
    u_n = ctx.field("u", 0, 1)
    
    # Apply boundary conditions
    zero = mod.cast(0)
    u_n = mod.where(iy == ny - 1, zero, u_n)
    
    # Compute derivatives
    u_xx = (u_e - 2*u + u_w) / (dx**2)
    u_yy = (u_n - 2*u + u_s) / (dy**2)
    
    # Define residual
    fu = u_xx + u_yy - source
    
    # Enforce BCs
    fu = mod.where(iy == ny - 1, (u - zero) / dx, fu)
    
    return [("fu", fu)]

# 2. PARSE ARGUMENTS
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=32)
    parser.add_argument("--Ny", type=int, default=32)
    odil.util.add_arguments(parser)
    return parser.parse_args()

# 3. MAKE PROBLEM
def make_problem(args):
    domain = odil.Domain(
        cshape=(args.Nx, args.Ny),
        dimnames=["x", "y"],
        lower=(0, 0),
        upper=(1, 1),
    )
    
    from odil import Field
    state = odil.State(
        fields={"u": Field(np.zeros(domain.size(loc="cc")), loc="cc")}
    )
    state = domain.init_state(state)
    
    extra = argparse.Namespace()
    extra.args = args
    
    problem = odil.Problem(operator, domain, extra)
    return problem, state

# 4. PLOT FUNCTION (optional)
def plot_func(problem, state, epoch, frame, cbinfo=None):
    # Plotting code
    pass

# 5. MAIN
def main():
    args = parse_args()
    odil.setup_outdir(args)
    problem, state = make_problem(args)
    callback = odil.make_callback(problem, args, plot_func=plot_func)
    odil.util.optimize(args, "adam", problem, state, callback)

if __name__ == "__main__":
    main()
```

## Common Patterns

### Boundary Conditions

```python
# Dirichlet BC: u = value at boundary
u_n = mod.where(iy == ny - 1, value, u_n)  # Apply to stencil
fu = mod.where(iy == ny - 1, (u - value) / dx, fu)  # Enforce in residual

# Neumann BC: du/dn = value at boundary
# Use extrapolation or ghost cells
```

### Finite Differences

```python
# First derivative (central)
u_x = (u_e - u_w) / (2 * dx)

# First derivative (upwind)
u_x = mod.where(u > 0, (u - u_w) / dx, (u_e - u) / dx)

# Second derivative
u_xx = (u_e - 2*u + u_w) / (dx**2)

# Laplacian (2D)
lap_u = (u_e - 2*u + u_w) / (dx**2) + (u_n - 2*u + u_s) / (dy**2)
```

### Multiple Equations

```python
def operator(ctx):
    # ... compute residuals for each equation ...
    fu = ...  # Equation 1
    fv = ...  # Equation 2
    fp = ...  # Equation 3
    
    return [("fu", fu), ("fv", fv), ("fp", fp)]
```

## Running Your Code

1. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run your script:**
   ```bash
   python your_script.py --Nx 64 --Ny 64 --epochs 1000
   ```

3. **Check output:**
   - `train.log` - Convergence log
   - `train.csv` - Training history
   - `field_*.png` - Plot files

## Tips

1. **Always call `domain.init_state(state)`** after creating state
2. **Use `ctx.mod` for math operations** (not numpy directly)
3. **Apply boundary conditions** to both stencil values and residuals
4. **Start simple** - test with known solutions first
5. **Use multigrid** for large problems
6. **Monitor convergence** in `train.log`

## Example: Lid-Driven Cavity

See `examples/lid_driven_cavity/lid_cavity.py` for a complete example solving the Navier-Stokes equations.

