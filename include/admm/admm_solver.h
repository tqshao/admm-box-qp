#pragma once

#include "types.h"
#include "riccati_solver.h"

#include <memory>

namespace admm {

// ADMM-based solver for constrained linear-quadratic trajectory optimization.
//
// The constructor performs the one-time offline work:
//   - LDLT path: Build the KKT matrix, apply Ruiz equilibration, factorize
//   - Riccati path: Allocate per-step gain matrices
//
// solve() runs the ADMM iterations. The y-update uses either:
//   - Sparse LDLT back-substitution (default), or
//   - Riccati backward-forward recursion (when use_riccati = true)
//
class ADMMSolver {
public:
    explicit ADMMSolver(const ProblemData& data);
    ADMMResult solve(const WarmStart& warm = WarmStart{}) const;
    double kkt_time_us() const { return kkt_time_us_; }

private:
    void buildBounds();
    void buildBaseKKT() const;           // KKT without rho*I (for Ruiz)
    void computeRuizScaling(int max_iter);
    void buildScaledKKT(double rho) const;  // scaled KKT with rho*I
    void refactorize(double new_rho) const;

    // Riccati y-update: extracts linear terms, runs backward+forward pass.
    Eigen::VectorXd riccatiYUpdate(double rho,
                                   const Eigen::VectorXd& z,
                                   const Eigen::VectorXd& lambda) const;

    // OSQP-style polishing: identify active set, solve reduced equality-constrained QP.
    // Modifies y in-place if polishing succeeds.
    void polishSolution(Eigen::VectorXd& y,
                        const Eigen::VectorXd& z,
                        const Eigen::VectorXd& lambda,
                        ADMMResult& result) const;

    ProblemData data_;
    int nx_, nu_, ny_, n_eq_;
    bool use_riccati_ = false;

    // --- LDLT path ---
    mutable Eigen::SparseMatrix<double> kkt_matrix_;
    mutable Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_;
    mutable Eigen::VectorXd d_;
    std::vector<int> diag_offsets_;
    mutable double current_rho_ = 0.0;

    // --- Ruiz scaling ---
    Eigen::VectorXd D_;                  // primal scaling  (size ny_)
    Eigen::VectorXd E_;                  // dual scaling    (size n_eq_)
    Eigen::VectorXd D_inv_;              // D^{-1}
    Eigen::VectorXd d_scaled_;           // E * d
    Eigen::VectorXd lower_bounds_scaled_;
    Eigen::VectorXd upper_bounds_scaled_;

    // --- Riccati path ---
    std::unique_ptr<RiccatiSolver> riccati_;

    Eigen::VectorXd lower_bounds_;
    Eigen::VectorXd upper_bounds_;
    mutable double kkt_time_us_ = 0.0;
};

}  // namespace admm
