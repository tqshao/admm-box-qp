#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <optional>
#include <vector>

namespace admm {

// Problem data for a linear-quadratic trajectory optimization with box constraints.
//
//   min  0.5 * sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + 0.5 * x_N^T P x_N
//   s.t. x_{k+1} = A x_k + B u_k,   k = 0, ..., N-1
//        x_min <= x_k <= x_max
//        u_min <= u_k <= u_max
//
struct ProblemData {
    // System dynamics: x_{k+1} = A x_k + B u_k
    Eigen::MatrixXd A;  // nx x nx
    Eigen::MatrixXd B;  // nx x nu

    // Stage cost matrices
    Eigen::MatrixXd Q;  // nx x nx
    Eigen::MatrixXd R;  // nu x nu
    Eigen::MatrixXd P;  // nx x nx  (terminal cost)

    // Box constraints (constant across the horizon)
    Eigen::VectorXd x_min, x_max;  // nx
    Eigen::VectorXd u_min, u_max;  // nu

    // Optional: per-element bounds overriding x_min/x_max/u_min/u_max.
    // Size must equal ny(). When set, buildBounds() uses these directly.
    // This enables time-varying constraints, e.g. obstacles at specific steps.
    std::optional<Eigen::VectorXd> custom_lower_bounds;
    std::optional<Eigen::VectorXd> custom_upper_bounds;

    int N = 0;  // horizon length

    // ADMM parameters
    double rho     = 1.0;     // penalty parameter
    double alpha   = 1.6;     // over-relaxation factor in [1.5, 1.8]
    double eps_abs = 1e-3;    // primal residual tolerance
    double eps_rel = 1e-3;   // dual residual tolerance
    int max_iter   = 1000;    // maximum ADMM iterations
    double kkt_reg = 1e-12;   // regularization for the (2,2) zero block
    bool   use_riccati = false; // use Riccati recursion instead of sparse LDLT for y-update

    // Adaptive rho parameters (OSQP-style)
    bool   adaptive_rho       = false;   // enable adaptive rho adjustment
    int    adapt_interval     = 25;      // check every N iterations
    double adapt_tolerance    = 5.0;     // minimum change ratio to trigger refactorization
    double rho_min            = 1e-6;    // lower bound on rho
    double rho_max            = 1e+6;    // upper bound on rho

    // Data-driven rho initialization (OSQP-style)
    bool   auto_rho           = false;   // compute initial rho from problem data

    // Polishing (OSQP-style)
    bool   polish             = true;    // enable solution polishing after ADMM convergence
    double polish_delta       = 1e-6;    // regularization for polishing KKT
    int    polish_refine_iter = 3;       // iterative refinement steps

    // Optional: fix initial state
    std::optional<Eigen::VectorXd> x0;

    // Dimension helpers
    int nx() const { return static_cast<int>(A.rows()); }
    int nu() const { return static_cast<int>(B.cols()); }

    // Total dimension of the stacked decision vector
    // y = [x0; u0; x1; u1; ...; x_{N-1}; u_{N-1}; x_N]
    int ny() const { return (N + 1) * nx() + N * nu(); }
};

// Optional warm-start data for the ADMM solver.
// Pass to solve() to initialize y, z, lambda from a previous solution.
struct WarmStart {
    Eigen::VectorXd y;      // primal variable (size ny)
    Eigen::VectorXd z;      // auxiliary variable (size ny)
    Eigen::VectorXd lambda; // dual variable (size ny)
};

struct ADMMResult {
    std::vector<Eigen::VectorXd> x;  // states  [x_0, ..., x_N]
    std::vector<Eigen::VectorXd> u;  // controls [u_0, ..., u_{N-1}]
    int iterations        = 0;
    double primal_residual = 0.0;
    double dual_residual   = 0.0;
    bool converged         = false;

    // Final rho value (useful when adaptive_rho is enabled)
    double final_rho = 0.0;

    // Polishing result
    bool   polished = false;        // true if polishing was attempted and accepted
    double time_polish_us = 0.0;    // polishing time (microseconds)

    // Final solution quality (after polishing if applied)
    double objective_cost    = 0.0;  // 0.5 * sum(x_k^T Q x_k + u_k^T R u_k) + 0.5 * x_N^T P x_N
    double max_bound_violation = 0.0; // max(0, max(|y_i - clip(y_i, l_i, u_i)|))

    // Pre-polish metrics (ADMM solution before polishing)
    double pre_polish_objective_cost    = 0.0;
    double pre_polish_max_bound_violation = 0.0;
    double pre_polish_primal_residual   = 0.0;
    double pre_polish_dual_residual     = 0.0;

    // Timing (microseconds)
    double time_kkt_us = 0.0;  // KKT build + factorization
    double time_solve_us = 0.0;  // ADMM iterations total
};

}  // namespace admm
