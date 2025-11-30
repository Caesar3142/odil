# Lid-Driven Cavity Flow

This example solves the steady-state incompressible Navier-Stokes equations for lid-driven cavity flow using the ODIL framework.

## Problem Description

The lid-driven cavity is a classic benchmark problem in computational fluid dynamics. A square cavity has three stationary walls and one moving wall (the "lid") that moves with constant velocity.

**Governing Equations:**
- Continuity: ∇·u = 0
- Momentum: (u·∇)u = -∇p + (1/Re)∇²u

**Boundary Conditions:**
- Top wall (y=1): u = 1, v = 0 (moving lid)
- Other walls: u = 0, v = 0 (no-slip)
- Pressure: Reference point at bottom-left corner (p = 0)

## How to Run

### Prerequisites

1. Activate the virtual environment:
   ```bash
   cd /Users/caesarwiratama/Documents_Local/GitHub/odil
   source venv/bin/activate
   ```

2. Navigate to the example directory:
   ```bash
   cd examples/lid_driven_cavity
   ```

### Running the Simulation

**Basic run (default: Re=100, 32x32 grid):**
```bash
python lid_cavity.py --Re 100 --Nx 32 --Ny 32 --epochs 2000
```

**Higher resolution:**
```bash
python lid_cavity.py --Re 100 --Nx 64 --Ny 64 --epochs 3000
```

**Higher Reynolds number:**
```bash
python lid_cavity.py --Re 400 --Nx 64 --Ny 64 --epochs 5000
```

**Different optimizer:**
```bash
python lid_cavity.py --Re 100 --optimizer lbfgs --lr 0.01
```

## Parameters

- `--Re`: Reynolds number (default: 100)
- `--Nx`: Grid size in x direction (default: 32)
- `--Ny`: Grid size in y direction (default: 32)
- `--epochs`: Number of optimization epochs (default: from plot_every * frames)
- `--optimizer`: Optimizer to use: "adam" or "lbfgs" (default: "adam")
- `--lr`: Learning rate (default: 1e-3)
- `--plot`: Enable plotting (default: 1)

## Output

The simulation creates output files in `out_lid_cavity/`:
- `train.log` - Training log with convergence information
- `train.csv` - Training history (loss, residuals, etc.)
- `field_*.png` - Plot files showing:
  - Velocity magnitude contours
  - Streamlines
  - Pressure contours

## Understanding the Code Structure

The code follows the standard ODIL template:

1. **`operator(ctx)`** - Defines the PDE as discrete residuals
   - Gets fields using `ctx.field("name")`
   - Computes derivatives using finite differences
   - Applies boundary conditions
   - Returns list of residuals: `[("name", residual), ...]`

2. **`make_problem(args)`** - Sets up the problem
   - Creates the domain (grid)
   - Initializes state variables (u, v, p)
   - Creates the Problem object

3. **`plot_func(...)`** - Visualizes results (optional)

4. **`main()`** - Runs the optimization

## Key ODIL Concepts Used

- **Field locations**: All fields are cell-centered (`loc="cc"`)
- **Boundary conditions**: Applied using `mod.where()` with indices
- **Finite differences**: Central differences for diffusion, upwind for advection
- **Residuals**: Each equation becomes a residual that should equal zero

## Tips

- Start with lower Re (e.g., 10-100) for faster convergence
- Use multigrid (`--multigrid 1`) for larger grids
- Adjust learning rate if convergence is slow
- Check `train.log` to monitor convergence

