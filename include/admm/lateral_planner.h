#pragma once

#include "admm_solver.h"

#include <string>

namespace admm {

// ADMM solver parameters. Defaults can be overridden per-scenario in scenarios.json.
struct SolverConfig {
    double rho      = 10.0;
    double alpha    = 1.6;
    double eps_abs  = 1e-3;
    double eps_rel = 1e-3;
    int    max_iter = 2000;
    double kkt_reg  = 1e-12;
    bool   use_riccati = false;

    // Adaptive rho (OSQP-style)
    bool   adaptive_rho       = false;
    int    adapt_interval     = 25;
    double adapt_tolerance    = 5.0;
    double rho_min            = 1e-6;
    double rho_max            = 1e+6;

    // Data-driven rho initialization (OSQP-style)
    bool   auto_rho           = false;

    // Polishing (OSQP-style)
    bool   polish             = true;
    double polish_delta       = 1e-6;
    int    polish_refine_iter = 3;
};

// Lateral planner parameters. Defaults can be overridden per-scenario in scenarios.json.
//
// Physical setup:
//   - Vehicle travels at constant longitudinal speed v_x.
//   - Lateral dynamics are modeled as a triple integrator:
//       state  x = [y, vy, ay]  (lateral position, velocity, acceleration)
//       input  u = [jerk]       (lateral jerk)
//   - Objective: track the lane centerline (y = 0) while staying inside
//     lane boundaries and respecting comfort limits.
//
struct PlannerConfig {
    // Vehicle / scenario
    double v_x             = 20.0;    // longitudinal speed [m/s]
    double dt              = 0.1;     // discretization time step [s]
    int    N               = 30;      // planning horizon (steps)

    // Lane geometry
    double lane_half_width = 1.75;    // half lane width [m]

    // Lateral limits
    double max_lat_vel     = 1.0;     // max lateral velocity [m/s]
    double max_lat_acc     = 3.0;     // max lateral acceleration [m/s²]
    double max_lat_jerk    = 10.0;    // max lateral jerk [m/s³]

    // Cost weights (stage)
    double q_y             = 5.0;     // lateral position tracking
    double q_vy            = 1.0;     // lateral velocity damping
    double q_ay            = 0.5;     // lateral acceleration smoothness
    double r               = 1.0;     // jerk penalty (comfort)

    // Cost weights (terminal)
    double p_y             = 50.0;    // terminal position
    double p_vy            = 10.0;    // terminal velocity
    double p_ay            = 5.0;     // terminal acceleration
};

// Load configs from JSON files.
SolverConfig  loadSolverConfig(const std::string& path);
PlannerConfig loadPlannerConfig(const std::string& path);

// Describes a rectangular obstacle occupying part of the lane.
struct ObstacleRegion {
    double y_lo;     // lower edge of obstacle [m]
    double y_hi;     // upper edge of obstacle [m]
    int    k_start;  // first time step
    int    k_end;    // last time step (inclusive)
};

// Wraps ADMMSolver for the lateral trajectory tracking problem.
// Uses triple-integrator dynamics: state = [y, vy, ay], input = [jerk].
//
class LateralPlanner {
public:
    LateralPlanner(const PlannerConfig& planner_cfg,
                   const SolverConfig& solver_cfg);

    // Compute optimal lateral trajectory from initial state.
    // x0 = [y0; vy0; ay0]  (lateral position, velocity, acceleration).
    ADMMResult plan(const Eigen::VectorXd& x0,
                    const WarmStart& warm = WarmStart{}) const;

    // Compute optimal lateral trajectory with obstacle avoidance.
    ADMMResult plan(const Eigen::VectorXd& x0,
                    const std::vector<ObstacleRegion>& obstacles,
                    const WarmStart& warm = WarmStart{}) const;

    const PlannerConfig& plannerConfig() const { return planner_cfg_; }
    const SolverConfig&  solverConfig()  const { return solver_cfg_; }
    const ProblemData&   problemData()   const { return problem_data_; }

    // Build per-element custom bounds that incorporate obstacles.
    std::pair<Eigen::VectorXd, Eigen::VectorXd>
    buildObstacleBounds(const std::vector<ObstacleRegion>& obstacles) const;

private:
    PlannerConfig planner_cfg_;
    SolverConfig  solver_cfg_;
    // Pre-filled ProblemData (everything except x0 and custom bounds).
    mutable ProblemData problem_data_;
};

}  // namespace admm
