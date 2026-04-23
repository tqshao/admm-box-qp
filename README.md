# ADMM Constrained Trajectory Optimization

C++ implementation of an ADMM-based solver for box-constrained linear-quadratic trajectory optimization, applied to autonomous driving lateral motion planning.

## Features

- **ADMM Solver** with two y-update backends:
  - **LDL^T path**: Sparse KKT factorization with Ruiz equilibration
  - **Riccati path**: O(N) backward-forward recursion exploiting block-bidiagonal structure
- **OSQP-style Ruiz equilibration**: diagonal scaling of problem data before KKT formation so ρ·I is a plain identity in scaled space
- **Solution polishing**: post-convergence active-set identification + reduced KKT solve for higher accuracy
- **Triple integrator** lateral dynamics: state = [y, vy, ay], input = [jerk]
- **Box constraints** on position, velocity, acceleration, and jerk
- **Obstacle avoidance** via time-varying constraint bounds
- **Adaptive rho** with tempered update (OSQP-style)
- **Over-relaxation** using ŷ in primal residual (OSQP convention)
- **Warm-start** interface for MPC receding-horizon applications
- **JSON configuration** for solver, planner, and scenario parameters
- **Python visualization** with 4-panel trajectory plots and timing analysis
- **OSQP benchmark**: optional side-by-side comparison against OSQP on identical problems

## Project Structure

```
include/admm/
  types.h             ProblemData, ADMMResult, WarmStart
  admm_solver.h       ADMMSolver (KKT + Riccati dual-path, Ruiz, polishing)
  riccati_solver.h    RiccatiSolver (backward-forward y-update)
  osqp_solver.h       OsqpSolver (benchmark wrapper, optional)
  lateral_planner.h   SolverConfig, PlannerConfig, LateralPlanner, ObstacleRegion
src/
  admm_solver.cpp     ADMM solver with Ruiz scaling, over-relaxation, adaptive rho, polishing
  riccati_solver.cpp  Riccati recursion with gain caching
  osqp_solver.cpp     OSQP C interface wrapper
  lateral_planner.cpp Lateral planner with triple integrator dynamics
apps/
  export_scenarios.cpp   Run all scenarios from JSON, export CSV + timing
  benchmark_solvers.cpp  ADMM (LDLT/Riccati) vs OSQP comparison
tests/
  test_admm.cpp          Unit tests for core solver
  test_lateral_planner.cpp  Unit tests for lateral planner
configs/
  scenarios.json          9 test scenarios with solver/planner overrides
scripts/
  plot_results.py         Generate PNG plots from CSV
```

## Build

Requirements: C++17 compiler (GCC 11+), CMake 3.14+, Eigen3, Python 3 with matplotlib/numpy.

```bash
# Configure
cmake -B build -DCMAKE_CXX_COMPILER=g++-11

# Build all targets
cmake --build build -j$(nproc)
```

Dependencies (Eigen3, nlohmann/json, Google Test, OSQP) are fetched automatically via CMake FetchContent.

## Usage

### Run Tests

```bash
cd build && ctest --output-on-failure
```

### Export Scenario Data

```bash
./build/export_scenarios figures configs/scenarios.json
```

This reads `configs/scenarios.json`, runs each scenario (with polishing enabled by default), and writes CSV files to `figures/csv/`.

### Generate Plots

```bash
MPLBACKEND=Agg python3 scripts/plot_results.py
```

Generates:
- `figures/png/<scenario>.png` — 4-panel trajectory plots (position, velocity, acceleration, jerk), with top-view for obstacle scenarios
- `figures/png/_timing_summary.png` — solver timing and iteration comparison chart

### Benchmark Against OSQP

```bash
./build/benchmark_solvers configs/scenarios.json
```

Runs ADMM-LDLT, ADMM-Riccati, and OSQP on all scenarios, prints side-by-side comparison of iterations, solve time, and objective cost.

### Use as a Library

```cpp
#include "admm/lateral_planner.h"

admm::PlannerConfig pc;   // uses struct defaults
admm::SolverConfig  sc;
sc.adaptive_rho = true;   // enable adaptive penalty
sc.polish = true;         // enable solution polishing (default)

admm::LateralPlanner planner(pc, sc);

Eigen::VectorXd x0(3);
x0 << 0.5, 0.0, 0.0;    // y=0.5m offset, zero velocity and acceleration

// Without obstacles
auto result = planner.plan(x0);

// With obstacles
std::vector<admm::ObstacleRegion> obs = {{-0.5, 1.75, 8, 16}};
result = planner.plan(x0, obs);

// Check result
bool ok = result.converged;
bool polished = result.polished;  // true if polishing improved the solution
```

### Scenario Configuration

Scenarios are defined in `configs/scenarios.json`. Each entry overrides defaults:

```json
{
  "name": "my_scenario",
  "x0": [0.5, 0.0, 0.0],
  "planner": {
    "N": 80,
    "lane_half_width": 1.75
  },
  "solver": {
    "rho": 10.0,
    "adaptive_rho": true,
    "polish": true,
    "max_iter": 3000
  },
  "obstacles": [
    { "y_lo": -0.5, "y_hi": 1.75, "k_start": 10, "k_end": 20 }
  ]
}
```

**Solver parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rho` | 10.0 | ADMM penalty parameter |
| `alpha` | 1.6 | Over-relaxation factor ∈ [1.5, 1.8] |
| `eps_abs` | 1e-3 | Absolute tolerance |
| `eps_rel` | 1e-3 | Relative tolerance |
| `max_iter` | 2000 | Maximum ADMM iterations |
| `adaptive_rho` | false | Enable OSQP-style adaptive rho |
| `adapt_interval` | 25 | Check interval for rho adaptation |
| `adapt_tolerance` | 5.0 | Minimum change ratio to trigger refactorization |
| `use_riccati` | false | Use Riccati recursion instead of sparse LDLT |
| `polish` | true | Enable post-convergence solution polishing |
| `polish_delta` | 1e-6 | Regularization for polishing KKT |
| `polish_refine_iter` | 3 | Iterative refinement steps in polishing |

**Planner parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.1 | Time step [s] |
| `N` | 80 | Horizon length (steps) |
| `lane_half_width` | 1.75 | Half lane width [m] |
| `max_lat_vel` | 1.5 | Max lateral velocity [m/s] |
| `max_lat_acc` | 3.0 | Max lateral acceleration [m/s²] |
| `max_lat_jerk` | 10.0 | Max lateral jerk [m/s³] |

---

# Algorithm

## 1. Problem Definition

$$\min_{\mathbf{x, u}} \quad \frac{1}{2} \sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + \frac{1}{2} x_N^T P x_N$$

subject to:
1. **Linear Dynamics:** $x_{k+1} = A x_k + B u_k, \quad k=0, \dots, N-1$
2. **Box Constraints:** $l \le y \le u$ where $y = [x_0, u_0, \dots, x_N]^T$

## 2. ADMM Variable Splitting

The original constrained problem is split by introducing an auxiliary variable $z$ of the same dimension as $y$:

$$\min_{y, z} \underbrace{\frac{1}{2} y^T H y + \mathbb{I}_{\{Cy = d\}}(y)}_{f(y)} + \underbrace{\mathbb{I}_{\{l \le z \le u\}}(z)}_{g(z)} \quad \text{s.t. } y = z$$

where:

- **$y$ (Physics Expert):** Stacked trajectory vector $y = [x_0; u_0; x_1; u_1; \dots; x_{N-1}; u_{N-1}; x_N]$ responsible for satisfying dynamics ($Cy = d$) and minimizing the LQR cost ($\frac{1}{2} y^T H y$)
- **$z$ (Boundary Expert):** Auxiliary variable responsible for satisfying box constraints ($l \le z \le u$)
- **$H$**: Block-diagonal Hessian $\mathrm{diag}(Q, R, \dots, Q, R, P)$
- **$C, d$**: Dynamics constraint matrix and RHS ($d = [0; \dots; 0; \bar{x}_0]$)
- **$\mathbb{I}_{\mathcal{S}}$**: Indicator function — 0 if in set $\mathcal{S}$, $+\infty$ otherwise

The **augmented Lagrangian** for the consensus problem $y = z$ is:

$$\mathcal{L}_\rho(y, z, \lambda) = f(y) + g(z) + \lambda^T(y - z) + \frac{\rho}{2}\|y - z\|^2$$

ADMM alternates between minimizing over $y$ (with dynamics), minimizing over $z$ (with bounds), and updating the dual variable $\lambda$.

## 3. ADMM Iterations

### Step 1: $y$-Update (KKT or Riccati)

$$y^{k+1} = \arg\min_y \frac{1}{2}y^T (\mathbf{H} + \rho \mathbf{I})\, y - y^T(\rho\, z^k - \lambda^k) \quad \text{s.t. } \mathbf{C}y = \mathbf{d}$$

KKT system:

$$\begin{bmatrix} \mathbf{H} + \rho \mathbf{I} & \mathbf{C}^T \\ \mathbf{C} & -\varepsilon \mathbf{I} \end{bmatrix} \begin{bmatrix} y^{k+1} \\ \nu \end{bmatrix} = \begin{bmatrix} \rho z^k - \lambda^k \\ \mathbf{d} \end{bmatrix}$$

**Two solver backends:**
- **LDLT**: Sparse factorization (with Ruiz equilibration) of the full KKT matrix. Factorize once, back-substitute each iteration.
- **Riccati**: Exploit block-bidiagonal structure for O(N·nx³) backward-forward recursion.

### Step 2: $z$-Update (Element-wise Projection)

$$z^{k+1} = \Pi_{[l,u]}(\hat{y}^{k+1} + \lambda^k / \rho)$$

where $\hat{y} = \alpha y + (1-\alpha) z$ is the over-relaxed variable.

### Step 3: $\lambda$-Update (Dual Ascent)

$$\lambda^{k+1} = \lambda^k + \rho (\hat{y}^{k+1} - z^{k+1})$$

### Convergence (OSQP Convention)

- **Primal:** $\|\hat{y} - z\|_\infty \le \epsilon_\text{abs} + \epsilon_\text{rel} \max(\|\hat{y}\|_\infty, \|z\|_\infty)$
- **Dual:** $\|\rho(z - z^\text{prev})\|_\infty \le \epsilon_\text{abs} + \epsilon_\text{rel} \|\lambda\|_\infty$

---

## 4. Ruiz Equilibration (LDLT Path)

Before forming the KKT matrix, compute diagonal scaling vectors $D$ (primal, size $n_y$) and $E$ (dual, size $n_\text{eq}$) via iterative equilibration on the base KKT matrix (without $\rho I$):

1. Build base KKT: $K_0 = \begin{bmatrix} H & C^T \\ C & -\varepsilon I \end{bmatrix}$

2. For $i = 1, \dots, 10$: compute per-element scaling $\delta_j = 1/\sqrt{\max(\|K_{i-1,j:}\|_\infty, \|K_{i-1,:j}\|_\infty)}$, then $K_i = \text{diag}(\delta) K_{i-1} \text{diag}(\delta)$

3. Accumulated scaling: $D = \prod \delta[0:n_y]$, $E = \prod \delta[n_y:]$

4. Build scaled KKT: $\begin{bmatrix} DHD + \rho I & (ECD)^T \\ ECD & -E\varepsilon E \end{bmatrix}$

The key insight: $\rho I$ is a **plain diagonal** in the scaled space (not conjugated by $D^2$), making $\rho$ tuning unnecessary.

---

## 5. Adaptive Rho

When enabled, adjust $\rho$ every `adapt_interval` iterations:

1. **Estimate:** $\hat\rho = \sqrt{r_\text{prim} / r_\text{dual}}$
2. **Temper:** $\rho_\text{new} = \rho \cdot \hat\rho^{0.3}$
3. **Clamp:** $\rho_\text{new} \in [\rho_\text{min}, \rho_\text{max}]$
4. **Gate:** only apply if $|\rho_\text{new}/\rho| > $ `adapt_tolerance`

The tempered update ($\hat\rho^{0.3}$ instead of $\hat\rho$) prevents oscillation.

---

## 6. Solution Polishing (OSQP-Style)

After ADMM converges, a polishing step attempts to recover a **higher-accuracy solution** by identifying the active constraint set and solving a single reduced equality-constrained QP.

### 6.1 Active Set Identification

For each element $i$ of the stacked vector $y$, examine the auxiliary variable $z$ and dual variable $\lambda$:

- **Lower-active:** $z_i - l_i < -\lambda_i$ (small gap + negative dual)
- **Upper-active:** $u_i - z_i < \lambda_i$ (small gap + positive dual)
- **Equality:** $l_i = u_i$

This criterion comes from the KKT complementary slackness: when a constraint is active, the gap between $z$ and its bound is small while the dual has the appropriate sign.

### 6.2 Reduced KKT System

Let $\mathcal{A}$ be the set of active bound indices with values $b_\mathcal{A}$. The polishing QP fixes active bounds as equalities:

$$\min_y \frac{1}{2} y^T H y \quad \text{s.t. } Cy = d, \quad y_\mathcal{A} = b_\mathcal{A}$$

With regularization $\delta$ for numerical stability, the polishing KKT is:

$$\begin{bmatrix} H + \delta I & C^T & S^T \\ C & -\varepsilon I & 0 \\ S & 0 & -\delta I \end{bmatrix} \begin{bmatrix} y \\ \nu \\ \mu \end{bmatrix} = \begin{bmatrix} \delta \cdot y_\text{ADMM} \\ d \\ b_\mathcal{A} \end{bmatrix}$$

where $S$ is the $|\mathcal{A}| \times n_y$ selection matrix for active constraints.

### 6.3 Iterative Refinement

The regularization $\delta$ introduces a small perturbation. Iterative refinement removes it:

$$\text{For } r = 1, \dots, R: \quad \Delta z = K_\text{reg}^{-1}(b_\text{true} - K_\text{true} z), \quad z \leftarrow z + \Delta z$$

### 6.4 Acceptance

The polished solution is accepted only if it is feasible (all bounds satisfied within tolerance). Otherwise, the ADMM solution is kept unchanged.

---

## 7. Riccati Y-Update

When `use_riccati = true`, the y-update exploits the block-bidiagonal KKT structure for O(N·nx³) instead of a full sparse factorization:

**Backward pass** (k = N-1 → 0):
$$S_N = P + \rho I, \quad \Sigma_k = R + \rho I + B^T S_{k+1} B$$
$$K_k = \Sigma_k^{-1} B^T S_{k+1}, \quad S_k = Q + \rho I + A^T(S_{k+1} - S_{k+1} B \Sigma_k^{-1} B^T S_{k+1}) A$$

**Forward pass** (k = 0 → N-1):
$$u_k = -K_k x_k - k^\text{ff}_k, \quad x_{k+1} = A x_k + B u_k$$

Gain matrices $K_k$, $S_k$ are cached and only recomputed when $\rho$ changes. Linear terms $k^\text{ff}_k$ are updated every iteration.

---

## 8. Performance Summary

Typical results on 9 scenarios (N=80, nx=3, nu=1):

| Scenario | LDLT Iters | Polished | Total Time |
|----------|-----------|----------|------------|
| 01 Lane keeping center | 1 | - | 0.5 ms |
| 02 Lane keeping offset | 18 | - | 0.4 ms |
| 03 Near boundary | 26 | - | 0.6 ms |
| 04 Offset + velocity | 16 | - | 0.4 ms |
| 05 Active avoidance | 26 | - | 0.6 ms |
| 06 Swerve left | 112 | Yes | 1.6 ms |
| 07 Swerve right | 112 | Yes | 1.6 ms |
| 08 S-curve | 442 | - | 4.2 ms |
| 09 Narrow gap | 111 | Yes | 1.6 ms |
