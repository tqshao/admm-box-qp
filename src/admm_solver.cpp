#include "admm/admm_solver.h"

#include <Eigen/SparseCholesky>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace admm {

// ---------------------------------------------------------------------------
// Constructor: build matrices, factorize once
// ---------------------------------------------------------------------------
ADMMSolver::ADMMSolver(const ProblemData& data) : data_(data) {
    nx_ = data_.nx();
    nu_ = data_.nu();
    ny_ = data_.ny();

    const bool has_x0 = data_.x0.has_value();
    const int n_dyn   = data_.N * nx_;
    n_eq_             = n_dyn + (has_x0 ? nx_ : 0);

    buildBounds();
    buildKKTMatrix(data_.rho);

    auto t0 = std::chrono::high_resolution_clock::now();
    solver_.analyzePattern(kkt_matrix_);
    solver_.factorize(kkt_matrix_);
    auto t1 = std::chrono::high_resolution_clock::now();
    kkt_time_us_ = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("ADMMSolver: KKT factorization failed");
    }
}

// ---------------------------------------------------------------------------
// Build per-element box bounds for the stacked vector y
// ---------------------------------------------------------------------------
void ADMMSolver::buildBounds() {
    lower_bounds_.resize(ny_);
    upper_bounds_.resize(ny_);

    int idx = 0;
    for (int k = 0; k < data_.N; ++k) {
        lower_bounds_.segment(idx, nx_) = data_.x_min;
        upper_bounds_.segment(idx, nx_) = data_.x_max;
        idx += nx_;

        lower_bounds_.segment(idx, nu_) = data_.u_min;
        upper_bounds_.segment(idx, nu_) = data_.u_max;
        idx += nu_;
    }
    lower_bounds_.segment(idx, nx_) = data_.x_min;
    upper_bounds_.segment(idx, nx_) = data_.x_max;

    if (data_.custom_lower_bounds.has_value()) {
        lower_bounds_ = *data_.custom_lower_bounds;
    }
    if (data_.custom_upper_bounds.has_value()) {
        upper_bounds_ = *data_.custom_upper_bounds;
    }
}

// ---------------------------------------------------------------------------
// Build the KKT matrix with a given rho
//
//   [H + rho*I   C^T ]
//   [C          -eps*I]
//
// ---------------------------------------------------------------------------
void ADMMSolver::buildKKTMatrix(double rho) const {
    const int dim = ny_ + n_eq_;
    const bool has_x0 = data_.x0.has_value();
    const int n_dyn   = data_.N * nx_;

    // Equality constraint RHS: C y = d
    d_.resize(n_eq_);
    d_.setZero();
    if (has_x0) {
        d_.tail(nx_) = *data_.x0;
    }

    const int nnz_estimate =
        data_.N * (nx_ * nx_ + nu_ * nu_) + nx_ * nx_
        + ny_
        + 2 * data_.N * (nx_ * nx_ + nx_ * nu_ + nx_)
        + n_eq_
        + (has_x0 ? 2 * nx_ : 0);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz_estimate);

    // --- Top-left block: H + rho*I ---
    int idx = 0;
    for (int k = 0; k < data_.N; ++k) {
        for (int i = 0; i < nx_; ++i)
            for (int j = 0; j < nx_; ++j)
                triplets.emplace_back(idx + i, idx + j, data_.Q(i, j));
        idx += nx_;

        for (int i = 0; i < nu_; ++i)
            for (int j = 0; j < nu_; ++j)
                triplets.emplace_back(idx + i, idx + j, data_.R(i, j));
        idx += nu_;
    }
    for (int i = 0; i < nx_; ++i)
        for (int j = 0; j < nx_; ++j)
            triplets.emplace_back(idx + i, idx + j, data_.P(i, j));

    // rho * I diagonal
    for (int i = 0; i < ny_; ++i)
        triplets.emplace_back(i, i, rho);

    // --- Dynamics constraints C and C^T ---
    const int stride = nx_ + nu_;

    for (int k = 0; k < data_.N; ++k) {
        const int row     = k * nx_;
        const int col_xk  = k * stride;
        const int col_uk  = k * stride + nx_;
        const int col_xk1 = (k + 1) * stride;

        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < nx_; ++j) {
                const double val = -data_.A(i, j);
                triplets.emplace_back(ny_ + row + i, col_xk + j, val);
                triplets.emplace_back(col_xk + j, ny_ + row + i, val);
            }
            for (int j = 0; j < nu_; ++j) {
                const double val = -data_.B(i, j);
                triplets.emplace_back(ny_ + row + i, col_uk + j, val);
                triplets.emplace_back(col_uk + j, ny_ + row + i, val);
            }
            triplets.emplace_back(ny_ + row + i, col_xk1 + i, 1.0);
            triplets.emplace_back(col_xk1 + i, ny_ + row + i, 1.0);
        }
    }

    // --- Initial state constraint x_0 = x0 ---
    if (has_x0) {
        const int row0 = n_dyn;
        for (int i = 0; i < nx_; ++i) {
            triplets.emplace_back(ny_ + row0 + i, i, 1.0);
            triplets.emplace_back(i, ny_ + row0 + i, 1.0);
        }
    }

    // --- Bottom-right block: -kkt_reg * I ---
    for (int i = 0; i < n_eq_; ++i)
        triplets.emplace_back(ny_ + i, ny_ + i, -data_.kkt_reg);

    kkt_matrix_.resize(dim, dim);
    kkt_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    kkt_matrix_.makeCompressed();
}

// ---------------------------------------------------------------------------
// Rebuild KKT with a new rho and re-factorize (for adaptive rho)
// ---------------------------------------------------------------------------
void ADMMSolver::refactorize(double new_rho) const {
    buildKKTMatrix(new_rho);
    solver_.analyzePattern(kkt_matrix_);
    solver_.factorize(kkt_matrix_);

    if (solver_.info() != Eigen::Success) {
        // Silently ignore — the next solve will also fail and we'll bail out.
    }
}

// ---------------------------------------------------------------------------
// ADMM iterations with warm-start and adaptive rho
// ---------------------------------------------------------------------------
ADMMResult ADMMSolver::solve(const WarmStart& warm) const {
    ADMMResult result;
    result.time_kkt_us = kkt_time_us_;

    auto solve_start = std::chrono::high_resolution_clock::now();

    // Initialize variables — warm-start if provided, else zero
    Eigen::VectorXd y, z, z_prev, lambda;

    if (warm.y.size() == ny_ && warm.z.size() == ny_ && warm.lambda.size() == ny_) {
        y      = warm.y;
        z      = warm.z;
        lambda = warm.lambda;
    } else {
        y      = Eigen::VectorXd::Zero(ny_);
        z      = Eigen::VectorXd::Zero(ny_);
        lambda = Eigen::VectorXd::Zero(ny_);
    }
    z_prev = z;

    double rho = data_.rho;

    Eigen::VectorXd rhs(ny_ + n_eq_);
    rhs.tail(n_eq_) = d_;

    for (int iter = 0; iter < data_.max_iter; ++iter) {
        // --- Step 1: y-update (KKT solve) ---
        rhs.head(ny_) = rho * z - lambda;

        Eigen::VectorXd sol = solver_.solve(rhs);
        if (solver_.info() != Eigen::Success) {
            result.converged = false;
            result.iterations = iter;
            result.final_rho = rho;
            break;
        }
        y = sol.head(ny_);

        // Over-relaxed y
        Eigen::VectorXd y_hat = data_.alpha * y + (1.0 - data_.alpha) * z;

        // --- Step 2: z-update (box projection) ---
        z_prev = z;
        Eigen::VectorXd v = y_hat + lambda / rho;
        z = v.cwiseMax(lower_bounds_).cwiseMin(upper_bounds_);

        // --- Step 3: lambda-update (dual ascent) ---
        lambda.noalias() += rho * (y_hat - z);

        // --- Stopping criteria ---
        double primal_res = (y - z).lpNorm<Eigen::Infinity>();
        double dual_res   = (rho * (z - z_prev)).lpNorm<Eigen::Infinity>();

        result.iterations      = iter + 1;
        result.primal_residual = primal_res;
        result.dual_residual   = dual_res;

        if (primal_res <= data_.eps_pri && dual_res <= data_.eps_dual) {
            result.converged = true;
            result.final_rho = rho;
            break;
        }

        // --- Adaptive rho ---
        if (data_.adaptive_rho && (iter + 1) % data_.adapt_interval == 0) {
            const double mu = data_.adapt_ratio;
            const double tau = data_.adapt_factor;

            if (primal_res > mu * dual_res) {
                rho *= tau;
                lambda /= tau;
                refactorize(rho);
            } else if (dual_res > mu * primal_res) {
                rho /= tau;
                lambda *= tau;
                refactorize(rho);
            }
        }
    }

    auto solve_end = std::chrono::high_resolution_clock::now();
    result.time_solve_us =
        std::chrono::duration<double, std::micro>(solve_end - solve_start).count();

    // Extract trajectory from y
    result.x.resize(data_.N + 1);
    result.u.resize(data_.N);
    int idx = 0;
    for (int k = 0; k < data_.N; ++k) {
        result.x[k] = y.segment(idx, nx_);
        idx += nx_;
        result.u[k] = y.segment(idx, nu_);
        idx += nu_;
    }
    result.x[data_.N] = y.segment(idx, nx_);

    result.final_rho = rho;
    return result;
}

}  // namespace admm
