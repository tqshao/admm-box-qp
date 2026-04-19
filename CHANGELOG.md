# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
