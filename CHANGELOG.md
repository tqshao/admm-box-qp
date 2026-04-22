# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-04-22

### Added

- Riccati backward-forward solver as alternative y-update path (O(N·nx³) per iteration instead of sparse LDLT)
- OSQP-style Ruiz equilibration: diagonal scaling of problem data (H, C, bounds) before forming the KKT matrix so that ρ·I is a plain identity in the scaled space
- Per-iteration convergence check (previously checked every 25 iterations)
- OSQP benchmark integration: optional `benchmark_solvers` app comparing ADMM solver against OSQP on identical problems
- OSQP solver wrapper (`osqp_solver.h/cpp`) converting ProblemData to OSQP's C interface
- `use_riccati` flag in SolverConfig / ProblemData to select Riccati or LDLT path
- Riccati caching: gain matrices recomputed only when ρ changes, linear terms updated every iteration

### Changed

- Over-relaxation residual now uses ŷ (OSQP convention): `pri_res = ‖ŷ − z‖` instead of `‖y − z‖`
- Adaptive ρ uses tempered update: `ρ_new = ρ · (ρ_estimate)^0.3` preventing oscillation
- eps_rel tightened from 1e-3 to 0.001 for all scenarios
- Convergence checked every iteration (adapt_interval still controls ρ adjustment frequency)
- Config files consolidated: removed separate `solver_config.json` and `planner_config.json`, all defaults and per-scenario overrides live in `scenarios.json`
- README expanded with detailed ADMM derivation (augmented Lagrangian, KKT system, convergence criteria)

### Fixed

- All 9 scenarios now converge, including previously failing scenarios:
  - 05_active_boundary_avoidance (was infeasible with vy₀ > max): 3000 → 26 iterations
  - 08_s_curve (dual residual explosion): 3000 → 442 iterations
- Scenarios 02–04 improved 10–15× in iteration count due to Ruiz preconditioning

## [1.0.0] - 2025-04-19

### Added

- ADMM solver with KKT pre-factorization (LDL^T sparse factorization) for box-constrained LQ trajectory optimization
- Over-relaxation support (α ∈ [1.5, 1.8])
- Adaptive rho strategy — automatically tunes penalty parameter based on primal/dual residual ratio
- Warm-start interface for MPC receding-horizon applications
- Lateral trajectory planner with triple integrator dynamics (state=[y, vy, ay], input=[jerk])
- Box constraints on position, velocity, acceleration, and jerk
- Obstacle avoidance via time-varying custom bounds
- JSON configuration for solver parameters, planner parameters, and scenario definitions
- Scenario runner (`export_scenarios`) loading from `configs/scenarios.json`, exporting CSV
- Python visualization script with 4-panel trajectory plots (position, velocity, acceleration, jerk)
- Top-view obstacle overlay for collision avoidance scenarios
- Timing summary bar chart for solver performance comparison
- 9 test scenarios: lane keeping, near boundary, active avoidance, swerve left/right, S-curve, narrow gap
- Unit tests for core ADMM solver and lateral planner (Google Test)
- MIT License
