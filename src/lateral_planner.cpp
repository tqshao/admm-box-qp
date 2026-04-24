#include "admm/lateral_planner.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace admm {

// ---------------------------------------------------------------------------
// JSON loaders
// ---------------------------------------------------------------------------
SolverConfig loadSolverConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: cannot open " << path << ", using defaults\n";
        return {};
    }
    nlohmann::json j = nlohmann::json::parse(f);
    SolverConfig cfg;
    if (j.contains("rho"))      cfg.rho      = j["rho"];
    if (j.contains("alpha"))    cfg.alpha    = j["alpha"];
    if (j.contains("eps_abs"))  cfg.eps_abs  = j["eps_abs"];
    if (j.contains("eps_rel")) cfg.eps_rel = j["eps_rel"];
    if (j.contains("max_iter")) cfg.max_iter = j["max_iter"];
    if (j.contains("kkt_reg"))  cfg.kkt_reg  = j["kkt_reg"];
    return cfg;
}

PlannerConfig loadPlannerConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Warning: cannot open " << path << ", using defaults\n";
        return {};
    }
    nlohmann::json j = nlohmann::json::parse(f);
    PlannerConfig cfg;
    if (j.contains("v_x"))             cfg.v_x             = j["v_x"];
    if (j.contains("dt"))              cfg.dt              = j["dt"];
    if (j.contains("N"))               cfg.N               = j["N"];
    if (j.contains("lane_half_width")) cfg.lane_half_width = j["lane_half_width"];
    if (j.contains("max_lat_vel"))     cfg.max_lat_vel     = j["max_lat_vel"];
    if (j.contains("max_lat_acc"))     cfg.max_lat_acc     = j["max_lat_acc"];
    if (j.contains("max_lat_jerk"))    cfg.max_lat_jerk    = j["max_lat_jerk"];
    if (j.contains("q_y"))             cfg.q_y             = j["q_y"];
    if (j.contains("q_vy"))            cfg.q_vy            = j["q_vy"];
    if (j.contains("q_ay"))            cfg.q_ay            = j["q_ay"];
    if (j.contains("r"))               cfg.r               = j["r"];
    if (j.contains("p_y"))             cfg.p_y             = j["p_y"];
    if (j.contains("p_vy"))            cfg.p_vy            = j["p_vy"];
    if (j.contains("p_ay"))            cfg.p_ay            = j["p_ay"];
    return cfg;
}

// ---------------------------------------------------------------------------
// LateralPlanner
// ---------------------------------------------------------------------------
LateralPlanner::LateralPlanner(const PlannerConfig& planner_cfg,
                               const SolverConfig& solver_cfg)
    : planner_cfg_(planner_cfg), solver_cfg_(solver_cfg) {
    const double dt = planner_cfg_.dt;

    // --- Dynamics: triple integrator ---
    // x = [y; vy; ay],  u = [jerk]
    // y_{k+1}  = y_k  + vy_k * dt + 0.5 * ay_k * dt^2 + (1/6) * jerk_k * dt^3
    // vy_{k+1} = vy_k + ay_k * dt + 0.5 * jerk_k * dt^2
    // ay_{k+1} = ay_k + jerk_k * dt
    problem_data_.A = Eigen::MatrixXd::Identity(3, 3);
    problem_data_.A(0, 1) = dt;
    problem_data_.A(0, 2) = 0.5 * dt * dt;
    problem_data_.A(1, 2) = dt;

    problem_data_.B = Eigen::MatrixXd::Zero(3, 1);
    problem_data_.B(0, 0) = (1.0 / 6.0) * dt * dt * dt;
    problem_data_.B(1, 0) = 0.5 * dt * dt;
    problem_data_.B(2, 0) = dt;

    // --- Stage cost: Q = diag(q_y, q_vy, q_ay) ---
    problem_data_.Q = Eigen::MatrixXd::Zero(3, 3);
    problem_data_.Q(0, 0) = planner_cfg_.q_y;
    problem_data_.Q(1, 1) = planner_cfg_.q_vy;
    problem_data_.Q(2, 2) = planner_cfg_.q_ay;

    problem_data_.R = Eigen::MatrixXd::Identity(1, 1) * planner_cfg_.r;

    // --- Terminal cost: P = diag(p_y, p_vy, p_ay) ---
    problem_data_.P = Eigen::MatrixXd::Zero(3, 3);
    problem_data_.P(0, 0) = planner_cfg_.p_y;
    problem_data_.P(1, 1) = planner_cfg_.p_vy;
    problem_data_.P(2, 2) = planner_cfg_.p_ay;

    // --- Box constraints ---
    problem_data_.x_min = Eigen::VectorXd(3);
    problem_data_.x_min << -planner_cfg_.lane_half_width,
                           -planner_cfg_.max_lat_vel,
                           -planner_cfg_.max_lat_acc;

    problem_data_.x_max = Eigen::VectorXd(3);
    problem_data_.x_max << planner_cfg_.lane_half_width,
                           planner_cfg_.max_lat_vel,
                           planner_cfg_.max_lat_acc;

    problem_data_.u_min = Eigen::VectorXd(1);
    problem_data_.u_min << -planner_cfg_.max_lat_jerk;

    problem_data_.u_max = Eigen::VectorXd(1);
    problem_data_.u_max << planner_cfg_.max_lat_jerk;

    // --- Horizon ---
    problem_data_.N = planner_cfg_.N;

    // --- Solver parameters ---
    problem_data_.rho           = solver_cfg_.rho;
    problem_data_.alpha         = solver_cfg_.alpha;
    problem_data_.eps_abs       = solver_cfg_.eps_abs;
    problem_data_.eps_rel      = solver_cfg_.eps_rel;
    problem_data_.max_iter      = solver_cfg_.max_iter;
    problem_data_.kkt_reg       = solver_cfg_.kkt_reg;
    problem_data_.adaptive_rho  = solver_cfg_.adaptive_rho;
    problem_data_.adapt_interval = solver_cfg_.adapt_interval;
    problem_data_.adapt_tolerance = solver_cfg_.adapt_tolerance;
    problem_data_.rho_min       = solver_cfg_.rho_min;
    problem_data_.rho_max       = solver_cfg_.rho_max;
    problem_data_.auto_rho      = solver_cfg_.auto_rho;
    problem_data_.use_riccati   = solver_cfg_.use_riccati;
    problem_data_.polish        = solver_cfg_.polish;
    problem_data_.polish_delta  = solver_cfg_.polish_delta;
    problem_data_.polish_refine_iter = solver_cfg_.polish_refine_iter;
}

ADMMResult LateralPlanner::plan(const Eigen::VectorXd& x0,
                                const WarmStart& warm) const {
    problem_data_.x0 = x0;
    problem_data_.custom_lower_bounds.reset();
    problem_data_.custom_upper_bounds.reset();
    ADMMSolver solver(problem_data_);
    return solver.solve(warm);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
LateralPlanner::buildObstacleBounds(
    const std::vector<ObstacleRegion>& obstacles) const {

    const int nx = 3;
    const int nu = 1;
    const int stride = nx + nu;
    const int ny = problem_data_.ny();

    Eigen::VectorXd lo(ny);
    Eigen::VectorXd hi(ny);

    int idx = 0;
    for (int k = 0; k < planner_cfg_.N; ++k) {
        lo.segment(idx, nx) = problem_data_.x_min;
        hi.segment(idx, nx) = problem_data_.x_max;
        idx += nx;
        lo.segment(idx, nu) = problem_data_.u_min;
        hi.segment(idx, nu) = problem_data_.u_max;
        idx += nu;
    }
    lo.segment(idx, nx) = problem_data_.x_min;
    hi.segment(idx, nx) = problem_data_.x_max;

    const double hw = planner_cfg_.lane_half_width;
    for (const auto& obs : obstacles) {
        for (int k = obs.k_start; k <= obs.k_end; ++k) {
            int y_idx = (k < planner_cfg_.N) ? k * stride : planner_cfg_.N * stride;

            if (obs.y_hi >= hw - 1e-9) {
                hi(y_idx) = std::min(hi(y_idx), obs.y_lo);
            }
            if (obs.y_lo <= -hw + 1e-9) {
                lo(y_idx) = std::max(lo(y_idx), obs.y_hi);
            }
        }
    }

    return {lo, hi};
}

ADMMResult LateralPlanner::plan(
    const Eigen::VectorXd& x0,
    const std::vector<ObstacleRegion>& obstacles,
    const WarmStart& warm) const {

    auto [lo, hi] = buildObstacleBounds(obstacles);

    problem_data_.x0 = x0;
    problem_data_.custom_lower_bounds = lo;
    problem_data_.custom_upper_bounds = hi;

    ADMMSolver solver(problem_data_);
    return solver.solve(warm);
}

}  // namespace admm
