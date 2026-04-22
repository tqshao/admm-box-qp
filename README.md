# ADMM Constrained Trajectory Optimization

C++ implementation of an ADMM-based solver for box-constrained linear-quadratic trajectory optimization, applied to autonomous driving lateral motion planning.

## Features

- **ADMM Solver** with pre-factorized KKT system (LDL^T sparse factorization)
- **Triple integrator** lateral dynamics: state = [y, vy, ay], input = [jerk]
- **Box constraints** on position, velocity, acceleration, and jerk
- **Obstacle avoidance** via time-varying constraint bounds
- **Adaptive rho** — automatically tunes penalty parameter for faster convergence
- **Warm-start** interface for MPC online receding-horizon use
- **JSON configuration** for solver, planner, and scenario parameters
- **Python visualization** with 4-panel trajectory plots and timing analysis

## Project Structure

```
include/admm/       Core headers
  types.h             ProblemData, ADMMResult, WarmStart
  admm_solver.h       ADMMSolver class (KKT pre-factorization + ADMM loop)
  lateral_planner.h   SolverConfig, PlannerConfig, LateralPlanner, ObstacleRegion
src/
  admm_solver.cpp     ADMM solver implementation
  lateral_planner.cpp Lateral planner with triple integrator dynamics + JSON loader
apps/
  export_scenarios.cpp  Run all scenarios from JSON, export CSV
tests/
  test_admm.cpp         Unit tests for core solver
  test_lateral_planner.cpp  Unit tests for lateral planner
configs/
  scenarios.json         9 test scenario definitions (planner/solver overrides per scenario)
scripts/
  plot_results.py        Generate PNG plots from CSV
```

## Build

Requirements: C++17 compiler (GCC 11+), CMake 3.14+, Eigen3, Python 3 with matplotlib/numpy.

```bash
# Configure
cmake -B build -DCMAKE_CXX_COMPILER=g++-11

# Build all targets
cmake --build build -j$(nproc)
```

Dependencies (Eigen3, nlohmann/json, Google Test) are fetched automatically via CMake FetchContent.

## Usage

### Run Tests

```bash
cd build
ctest --output-on-failure
```

### Export Scenario Data

```bash
# Default: output to figures/
./build/export_scenarios figures

# Custom output dir and scenario config
./build/export_scenarios output_dir path/to/scenarios.json
```

This reads `configs/scenarios.json`, runs each scenario, and writes CSV files to `figures/csv/`.

### Generate Plots

```bash
cd /path/to/project
MPLBACKEND=Agg python3 scripts/plot_results.py
```

Generates:
- `figures/png/<scenario>.png` — 4-panel trajectory plots (position, velocity, acceleration, jerk), with top-view for obstacle scenarios
- `figures/png/_timing_summary.png` — solver timing and iteration comparison chart

### Use as a Library

```cpp
#include "admm/lateral_planner.h"

admm::PlannerConfig pc;   // uses struct defaults
admm::SolverConfig  sc;
sc.adaptive_rho = true;   // enable adaptive penalty

admm::LateralPlanner planner(pc, sc);

Eigen::VectorXd x0(3);
x0 << 0.5, 0.0, 0.0;    // y=0.5m offset, zero velocity and acceleration

// Without obstacles
auto result = planner.plan(x0);

// With obstacles
std::vector<admm::ObstacleRegion> obs = {{-0.5, 1.75, 8, 16}};
result = planner.plan(x0, obs);

// Warm-start from previous solution (for MPC)
admm::WarmStart warm;
// ... fill warm.y, warm.z, warm.lambda from previous ADMMResult ...
result = planner.plan(x0, obs, warm);
```

### Scenario Configuration

Scenarios are defined in `configs/scenarios.json`. Each entry overrides defaults:

```json
{
  "name": "my_scenario",
  "x0": [0.5, 0.0, 0.0],
  "planner": {
    "N": 30,
    "lane_half_width": 1.75,
    "max_lat_vel": 1.0,
    "max_lat_acc": 3.0,
    "max_lat_jerk": 10.0
  },
  "solver": {
    "adaptive_rho": true,
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
| `alpha` | 1.6 | Over-relaxation factor |
| `eps_pri` | 1e-3 | Primal residual tolerance |
| `eps_dual` | 1e-3 | Dual residual tolerance |
| `max_iter` | 2000 | Maximum ADMM iterations |
| `adaptive_rho` | false | Auto-tune rho for convergence |
| `adapt_interval` | 25 | Check interval for rho adaptation |
| `adapt_ratio` | 10.0 | Primal/dual ratio threshold |
| `adapt_factor` | 2.0 | Rho multiply/divide factor |

**Planner parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.1 | Time step [s] |
| `N` | 30 | Horizon length (steps) |
| `lane_half_width` | 1.75 | Half lane width [m] |
| `max_lat_vel` | 1.0 | Max lateral velocity [m/s] |
| `max_lat_acc` | 3.0 | Max lateral acceleration [m/s^2] |
| `max_lat_jerk` | 10.0 | Max lateral jerk [m/s^3] |

---

# Algorithm
## 1. Problem Definition

The goal is to solve a Linear-Quadratic Regulator (LQR) trajectory optimization problem over a horizon $N$, subject to both system dynamics and state/input box constraints.

### Objective Function (Cost)

$$\min_{\mathbf{x, u}} \quad \frac{1}{2} \sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + \frac{1}{2} x_N^T P x_N$$

### Constraints

1. **Linear Dynamics:** $x_{k+1} = A x_k + B u_k, \quad k=0, \dots, N-1$

2. **Boundary Constraints:** $x_{min} \le x_k \le x_{max}, \quad u_{min} \le u_k \le u_{max}$


---

## 2. ADMM Variable Splitting

To apply ADMM, we decouple the "coupled physics" from the "local boundaries" by splitting the variables into two sets:

- **Variable $\mathbf{y}$ (The "Physics" Expert):** Contains the full trajectory $[x_0, u_0, \dots, x_N]^T$. This variable is responsible for strictly satisfying the **system dynamics**.

- **Variable $\mathbf{z}$ (The "Boundary" Expert):** A vector of the same dimension as $y$. This variable is responsible for strictly staying within **box constraints**.


**Consensus Form:**

$$\min_{y, z} \quad f(y) + g(z) \quad \text{s.t. } y - z = 0$$

Where $f(y)$ handles the cost and dynamics, and $g(z)$ is the indicator function for the boundary constraints.

---

## 3. The Iterative Algorithm

Given a penalty parameter $\rho > 0$ and dual variables (multipliers) $\lambda$, perform the following three steps in each iteration:

### Step 1: $y$-Update (Solving the Global KKT System)

Starting from the **augmented Lagrangian** of the consensus problem:

$$\mathcal{L}_\rho(y, z, \lambda) = f(y) + g(z) + \lambda^T(y - z) + \frac{\rho}{2}\|y - z\|^2$$

The $y$-update fixes $z = z^k$ and $\lambda = \lambda^k$, and minimizes over $y$:

$$y^{k+1} = \arg\min_{y} \; f(y) + \lambda^{k,T}(y - z^k) + \frac{\rho}{2}\|y - z^k\|^2 \quad \text{s.t. } \mathbf{C}y = \mathbf{d}$$

Completing the square (absorbing constant terms w.r.t. $y$):

$$y^{k+1} = \arg\min_{y} \; f(y) + \frac{\rho}{2}\left\|y - z^k + \frac{\lambda^k}{\rho}\right\|^2 \quad \text{s.t. } \mathbf{C}y = \mathbf{d}$$

Since $f(y) = \frac{1}{2}y^T \mathbf{H} y$ is quadratic (the stacked LQR cost), this subproblem is an **equality-constrained QP**:

$$\min_{y} \quad \frac{1}{2}y^T (\mathbf{H} + \rho \mathbf{I})\, y - y^T(\rho\, z^k - \lambda^k) \quad \text{s.t. } \mathbf{C}y = \mathbf{d}$$

Introduce KKT multiplier $\nu$ for the equality constraint $\mathbf{C}y = \mathbf{d}$, the **first-order optimality conditions** are:

$$\frac{\partial}{\partial y}: \quad (\mathbf{H} + \rho \mathbf{I})\, y + \mathbf{C}^T \nu = \rho\, z^k - \lambda^k$$

$$\frac{\partial}{\partial \nu}: \quad \mathbf{C}y = \mathbf{d}$$

Written in **block matrix form**:

$$\begin{bmatrix} \mathbf{H} + \rho \mathbf{I} & \mathbf{C}^T \\ \mathbf{C} & \mathbf{0} \end{bmatrix} \begin{bmatrix} y^{k+1} \\ \nu \end{bmatrix} = \begin{bmatrix} \rho z^k - \lambda^k \\ \mathbf{d} \end{bmatrix}$$

**Matrix definitions:**

- **$\mathbf{H}$**: Block-diagonal Hessian of the LQR cost, composed of $Q, R, P$:

$$\mathbf{H} = \mathrm{diag}(\underbrace{Q, R, \dots, Q, R}_{N \text{ stages}}, P)$$

- **$\mathbf{C}, \mathbf{d}$**: Linear equality constraints encoding the **dynamics**. Each block-row $k$ enforces $x_{k+1} - A x_k - B u_k = d_k$. For the standard trajectory problem, $d_k = 0$ (homogeneous dynamics). If an initial state $\bar{x}_0$ is provided, additional rows enforce $x_0 = \bar{x}_0$, giving:

$$\mathbf{d} = \begin{bmatrix} \mathbf{0}_{N \cdot n_x} \\ \bar{x}_0 \end{bmatrix}$$

> **Key insight:** The left-hand side KKT matrix is **constant** across iterations (it depends only on $\mathbf{H}$, $\mathbf{C}$, and $\rho$). This allows **pre-factorization** — factorize once offline, and each iteration only requires a back-substitution with the updated right-hand side. 


### Step 2: $z$-Update (Element-wise Projection)

Find $z^{k+1}$ that satisfies the boundaries while staying close to $y^{k+1}$:

$$z^{k+1} = \Pi_{\mathcal{Z}} (y^{k+1} + \frac{\lambda^k}{\rho})$$

**Operation (Clipping):** For each element $v_i$ in the vector:

$$z_i^{k+1} = \max(bound_{min}, \min(bound_{max}, v_i))$$

### Step 3: $\lambda$-Update (Dual Ascent)

Update the "pressure" to enforce the $y=z$ agreement:

$$\lambda^{k+1} = \lambda^k + \rho (y^{k+1} - z^{k+1})$$

---

## 4. Implementation & Performance Optimization

### 4.1 Pre-factorization (The "Secret Sauce")

Since the left-hand side of the KKT matrix is constant (assuming constant $\rho$ and linear dynamics), you should **not** re-solve the matrix from scratch:

1. **Offline Phase:** Compute the **$LDL^T$ sparse factorization** of the KKT matrix once.

2. **Online Phase:** In each ADMM iteration, perform only **Back-substitution**. This reduces the per-iteration complexity from $O(N^3)$ to **$O(N)$**.


### 4.2 Over-relaxation

Accelerate convergence by introducing a relaxation factor $\alpha \in [1.5, 1.8]$. Replace $y^{k+1}$ in the $z$ and $\lambda$ updates with:

$$\hat{y}^{k+1} = \alpha y^{k+1} + (1 - \alpha) z^k$$

### 4.3 Adaptive Penalty ($\rho$)

When `adaptive_rho` is enabled, the solver automatically adjusts $\rho$ every `adapt_interval` iterations based on the primal/dual residual ratio:

- If primal residual $> \mu \times$ dual residual: $\rho \leftarrow \rho \times \tau$, re-factorize KKT
- If dual residual $> \mu \times$ primal residual: $\rho \leftarrow \rho / \tau$, re-factorize KKT

This eliminates the need for manual rho tuning across different scenarios.

### 4.4 Warm-Start

For MPC applications, the `WarmStart` struct allows initializing ADMM variables (y, z, lambda) from a previous solution. This can significantly reduce iteration count when consecutive planning problems are similar.

### 4.5 Stopping Criteria

Terminate the algorithm when both residuals are below your threshold:

- **Primal Residual:** $\|y^{k+1} - z^{k+1}\|_\infty \le \epsilon_{pri}$ (Ensures feasibility)

- **Dual Residual:** $\|\rho(z^{k+1} - z^k)\|_\infty \le \epsilon_{dual}$ (Ensures optimality)


> Note: Based on your requirement, $\epsilon \approx 10^{-3}$ is a solid engineering target.

---

## 5. Summary of the Solution

- **Pros:** Avoids complex active-set logic for inequality constraints; each iteration involves only basic linear algebra; highly parallelizable.

- **Best For:** Real-time MPC with heavy state/input box constraints.

- **Numerical Tip:** If the KKT matrix is ill-conditioned, add a tiny regularization term $-\epsilon \mathbf{I}$ (e.g., $10^{-12}$) to the bottom-right zero block.
