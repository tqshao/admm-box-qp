#pragma once

#include "types.h"

namespace admm {

// ADMM-based solver for constrained linear-quadratic trajectory optimization.
//
// The constructor performs the one-time offline work:
//   - Build the KKT matrix  [H+rho*I  C^T; C  -eps*I]
//   - Symbolic + numeric sparse LDL^T factorization
//
// solve() runs the ADMM iterations using only back-substitution per iteration.
//
class ADMMSolver {
public:
    // Constructs the solver and performs KKT pre-factorization.
    // Throws std::runtime_error if factorization fails.
    explicit ADMMSolver(const ProblemData& data);

    // Runs ADMM and returns the result (including timing).
    // Optionally pass warm-start data (y, z, lambda from a previous solve).
    ADMMResult solve(const WarmStart& warm = WarmStart{}) const;

    // Get KKT factorization time in microseconds (set after construction).
    double kkt_time_us() const { return kkt_time_us_; }

private:
    void buildBounds();
    void buildKKTMatrix(double rho) const;

    // Rebuild KKT with a new rho and re-factorize (for adaptive rho).
    void refactorize(double new_rho) const;

    ProblemData data_;
    int nx_;    // state dimension
    int nu_;    // control dimension
    int ny_;    // decision variable dimension
    int n_eq_;  // number of equality constraints

    // Using SimplicialLDLT for symmetric quasi-definite KKT system.
    // Mutable because adaptive rho triggers re-factorization inside solve().
    mutable Eigen::SparseMatrix<double> kkt_matrix_;
    mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_;
    mutable Eigen::VectorXd d_;  // RHS of equality constraints (rebuilt on refactorize)

    Eigen::VectorXd lower_bounds_;  // stacked box lower bounds (size ny)
    Eigen::VectorXd upper_bounds_;  // stacked box upper bounds (size ny)
    mutable double kkt_time_us_ = 0.0;     // KKT factorization time [us]
};

}  // namespace admm
