# Lid-Driven Cavity Flow - Setup Summary

## Overview

This implementation solves the steady-state incompressible Navier-Stokes equations for lid-driven cavity flow using the ODIL framework.

**Governing Equations:**
- Continuity: ∇·u = 0
- Momentum: (u·∇)u = -∇p + (1/Re)∇²u

Where:
- **u** = x-velocity (horizontal velocity component)
- **v** = y-velocity (vertical velocity component)
- **p** = pressure
- **Re** = Reynolds number

**Boundary Conditions:**
- Top wall (y=1): u = 1, v = 0 (moving lid)
- Other walls: u = 0, v = 0 (no-slip)
- Pressure: Reference point at bottom-left corner (p = 0)

## Prerequisites

### 1. Virtual Environment Setup

```bash
cd /Users/caesarwiratama/Documents_Local/GitHub/odil
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install "jax[cpu]"  # or tensorflow if preferred
```

### 2. Required Dependencies

- Python 3.12+
- numpy
- matplotlib
- scipy
- JAX or TensorFlow (backend for ODIL)
- odil (installed via `pip install -e .`)

## File Structure

```
lid_driven_cavity/
├── lid_cavity.py          # Main simulation script
├── README.md              # Detailed documentation
├── SETUP_SUMMARY.md       # This file
└── out_lid_cavity/        # Output directory (created automatically)
    ├── train.log          # Training log
    ├── train.csv          # Training history
    ├── field_*.png        # Field visualizations
    ├── centerlines_*.png  # Centerline velocity profiles
    ├── centerlines_*.csv  # Centerline data
    └── convergence.png    # Convergence plots
```

## Running the Simulation

### Basic Usage

```bash
cd examples/lid_driven_cavity
source ../../venv/bin/activate
python lid_cavity.py --Re 100 --Nx 32 --Ny 32 --epochs 2000
```

### Recommended Parameters

**For Re = 100 (moderate resolution):**
```bash
python lid_cavity.py --Re 100 --Nx 128 --Ny 128 --epochs 3000 --frames 15 --plot_every 200
```

**For Re = 400 (higher resolution, no multigrid):**
```bash
python lid_cavity.py --Re 400 --Nx 100 --Ny 100 --multigrid 0 --epochs 5000 --frames 20
```

**For Re = 400 (with multigrid, use power-of-2 grid):**
```bash
python lid_cavity.py --Re 400 --Nx 128 --Ny 128 --epochs 5000 --frames 20 --lr 5e-4
```

## Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--Re` | Reynolds number | 100 | Higher Re = more complex flow |
| `--Nx` | Grid size in x | 32 | Use power of 2 for multigrid |
| `--Ny` | Grid size in y | 32 | Use power of 2 for multigrid |
| `--epochs` | Optimization iterations | 2000 | More epochs = better convergence |
| `--optimizer` | Optimizer type | "adam" | Options: "adam", "lbfgs" |
| `--lr` | Learning rate | 1e-3 | Lower for higher Re |
| `--multigrid` | Enable multigrid | 1 | Requires power-of-2 grid size |
| `--frames` | Number of plot frames | 10 | Set to 0 to disable plots |
| `--plot_every` | Epochs between plots | 100 | |
| `--report_every` | Epochs between reports | 50 | |

## Grid Size Recommendations

### Power of 2 (enables multigrid, faster):
- **32×32** - Quick testing
- **64×64** - Good for Re ≤ 100
- **128×128** - Recommended for Re = 100-400
- **256×256** - High resolution, slower

### Non-power of 2 (disable multigrid):
- **100×100** - Use `--multigrid 0`
- **150×150** - Use `--multigrid 0`
- Any custom size with `--multigrid 0`

## Output Files

### 1. Field Visualizations (`field_*.png`)
- Velocity magnitude contours
- Velocity vector fields
- Pressure contours

### 2. Centerline Plots (`centerlines_*.png`)
- **Left plot**: u (x-velocity) vs y along vertical centerline (x=0.5)
- **Right plot**: v (y-velocity) vs x along horizontal centerline (y=0.5)

### 3. Centerline Data (`centerlines_*.csv`)
CSV format with columns:
- `y`: y-coordinates for vertical centerline
- `u_x_velocity`: x-velocity values
- `x`: x-coordinates for horizontal centerline
- `v_y_velocity`: y-velocity values

### 4. Convergence Plots (`convergence.png`)
- Residual norms (fu, fv, fdiv)
- Loss convergence
- Maximum velocity tracking
- Center point velocities

### 5. Training History (`train.csv`)
Contains:
- Epoch number
- Residual norms for each equation
- Loss values
- Maximum velocity
- Center velocities
- Memory usage
- Walltime

## Implementation Details

### Operator Function
The `operator(ctx)` function defines the Navier-Stokes equations as discrete residuals:
- Uses finite differences for derivatives
- Upwind scheme for advection terms
- Central differences for diffusion
- Boundary conditions enforced using `mod.where()` with indices

### Field Locations
All fields are **cell-centered** (`loc="cc"`):
- u: x-velocity at cell centers
- v: y-velocity at cell centers
- p: pressure at cell centers

### Boundary Conditions
Applied in two steps:
1. **Stencil values**: Correct neighboring values at boundaries
2. **Residuals**: Enforce boundary conditions in the residual equations

## Troubleshooting

### Multigrid Error
**Error**: `ValueError: Expected equal 'X' and 'Y' with cshapes=...`

**Solution**: Use power-of-2 grid size OR disable multigrid:
```bash
python lid_cavity.py --Nx 100 --Ny 100 --multigrid 0
```

### Slow Convergence
- Reduce learning rate: `--lr 5e-4`
- Use more epochs: `--epochs 5000`
- Try different optimizer: `--optimizer lbfgs`

### Memory Issues
- Reduce grid size
- Use float32: `--double 0` (default is float64)

### Backend Not Found
**Error**: `Cannot select a default backend`

**Solution**: Install JAX or TensorFlow:
```bash
pip install "jax[cpu]"  # or
pip install tensorflow
```

## Code Structure

```
lid_cavity.py
├── operator(ctx)          # Defines PDE as residuals
├── parse_args()           # Command-line arguments
├── plot_func()            # Field visualizations
├── plot_centerlines()     # Centerline plots and CSV
├── plot_error()           # Error tracking
├── plot_convergence()     # Convergence plots
├── make_problem()         # Domain and state setup
└── main()                 # Main execution
```

## Example Workflow

1. **Activate environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run simulation:**
   ```bash
   cd examples/lid_driven_cavity
   python lid_cavity.py --Re 100 --Nx 128 --Ny 128 --epochs 3000
   ```

3. **Monitor progress:**
   ```bash
   tail -f out_lid_cavity/train.log
   ```

4. **View results:**
   - Check `out_lid_cavity/` for plots and data
   - Open `convergence.png` for convergence analysis
   - Check `centerlines_*.csv` for centerline data

## Validation

For Re = 100, typical results:
- Maximum u-velocity at top wall: ~1.0
- Center u-velocity (x=0.5, y=0.5): ~0.2-0.3
- Characteristic recirculation zones in corners
- Smooth velocity profiles along centerlines

## References

- ODIL Framework: https://github.com/cselab/odil
- Lid-driven cavity benchmark: Classic CFD test case
- See `HOW_TO_USE_ODIL_TEMPLATE.md` for general ODIL usage

## Notes

- Multigrid significantly speeds up convergence for large grids
- Higher Re requires more epochs and potentially lower learning rate
- Centerline plots are generated at each plot frame
- Convergence plot is generated at the end of simulation
- All outputs are saved in `out_lid_cavity/` directory

