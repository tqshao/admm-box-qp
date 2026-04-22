#pragma once

#include "types.h"

namespace admm {

// Result from OSQP solver (mirrors ADMMResult for comparison).
struct OsqpResult {
    std::vector<Eigen::VectorXd> x;  // states  [x_0, ..., x_N]
    std::vector<Eigen::VectorXd> u;  // controls [u_0, ..., u_{N-1}]
    int iterations          = 0;
    double primal_residual   = 0.0;
    double dual_residual     = 0.0;
    bool converged           = false;

    // Timing (microseconds)
    double time_setup_us  = 0.0;  // matrix construction + OSQP setup
    double time_solve_us  = 0.0;  // OSQP solve call
};

// OSQP-based solver for constrained linear-quadratic trajectory optimization.
//
// Translates ProblemData into OSQP standard form:
//   min  0.5 z^T P z + q^T z
//   s.t. l <= A z <= u
//
// where z = y (stacked decision vector), P = H (cost Hessian),
// A = [C; I] (dynamics equalities + box bounds).
class OsqpSolver {
public:
    explicit OsqpSolver(const ProblemData& data);
    ~OsqpSolver();

    OsqpSolver(const OsqpSolver&) = delete;
    OsqpSolver& operator=(const OsqpSolver&) = delete;

    OsqpResult solve() const;

    double setupTimeUs() const { return setup_time_us_; }

private:
    ProblemData data_;
    int nx_, nu_, ny_, n_eq_, n_constr_;

    // Opaque pointer to OSQP workspace
    void* osqp_workspace_ = nullptr;

    double setup_time_us_ = 0.0;
};

}  // namespace admm
