#include "admm/admm_solver.h"

#include <Eigen/SparseCholesky>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace admm {

// ---------------------------------------------------------------------------
// Constructor: build matrices, equilibrate, factorize once
// ---------------------------------------------------------------------------
ADMMSolver::ADMMSolver(const ProblemData& data) : data_(data) {
    nx_ = data_.nx();
    nu_ = data_.nu();
    ny_ = data_.ny();

    const bool has_x0 = data_.x0.has_value();
    const int n_dyn   = data_.N * nx_;
    n_eq_             = n_dyn + (has_x0 ? nx_ : 0);

    buildBounds();
    use_riccati_ = data_.use_riccati;

    if (use_riccati_) {
        riccati_ = std::make_unique<RiccatiSolver>(nx_, nu_, data_.N);
        kkt_time_us_ = 0.0;
    } else {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Step 1: Build base KKT (without rho*I) for Ruiz computation
        buildBaseKKT();

        // Step 2: Compute Ruiz scaling from base KKT
        computeRuizScaling(10);

        // Step 3: Build scaled KKT with rho*I as plain diagonal
        buildScaledKKT(data_.rho);

        // Cache positions of top-left diagonal entries in valuePtr()
        {
            const int* innerIdx = kkt_matrix_.innerIndexPtr();
            const int* outerPtr = kkt_matrix_.outerIndexPtr();
            diag_offsets_.resize(ny_);
            for (int j = 0; j < ny_; ++j) {
                for (int k = outerPtr[j]; k < outerPtr[j + 1]; ++k) {
                    if (innerIdx[k] == j) {
                        diag_offsets_[j] = k;
                        break;
                    }
                }
            }
        }
        current_rho_ = data_.rho;

        solver_.analyzePattern(kkt_matrix_);
        solver_.factorize(kkt_matrix_);
        auto t1 = std::chrono::high_resolution_clock::now();
        kkt_time_us_ = std::chrono::duration<double, std::micro>(t1 - t0).count();

        if (solver_.info() != Eigen::Success) {
            throw std::runtime_error("ADMMSolver: KKT factorization failed");
        }
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
// Build base KKT matrix (without rho*I) for Ruiz equilibration
//
//   [H          C^T    ]
//   [C          -eps*I ]
//
// Also computes the equality constraint RHS d_.
// ---------------------------------------------------------------------------
void ADMMSolver::buildBaseKKT() const {
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
        + 2 * data_.N * (nx_ * nx_ + nx_ * nu_ + nx_)
        + n_eq_
        + (has_x0 ? 2 * nx_ : 0);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz_estimate);

    // --- Top-left block: H (NO rho*I) ---
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
// Ruiz equilibration: compute diagonal scaling D, E from the base KKT
//
// Iteratively scales rows/columns so that the matrix has ~unit row/col norms.
// D scales primal variables (size ny_), E scales dual variables (size n_eq_).
// ---------------------------------------------------------------------------
void ADMMSolver::computeRuizScaling(int max_iter) {
    const int dim = ny_ + n_eq_;

    // Cumulative scaling vector (partitioned as [D; E])
    Eigen::VectorXd cum = Eigen::VectorXd::Ones(dim);

    // Work on a copy of the base KKT
    Eigen::SparseMatrix<double> M = kkt_matrix_;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute row and column inf-norms
        Eigen::VectorXd row_inf = Eigen::VectorXd::Zero(dim);
        Eigen::VectorXd col_inf = Eigen::VectorXd::Zero(dim);

        for (int j = 0; j < M.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(M, j); it; ++it) {
                double abs_val = std::abs(it.value());
                row_inf(it.row()) = std::max(row_inf(it.row()), abs_val);
                col_inf(j) = std::max(col_inf(j), abs_val);
            }
        }

        // Per-element scaling: 1 / sqrt(max(row_norm, col_norm))
        Eigen::VectorXd delta(dim);
        for (int i = 0; i < dim; ++i) {
            delta(i) = 1.0 / std::sqrt(std::max(row_inf(i), col_inf(i)));
        }

        // Apply: M = diag(delta) * M * diag(delta)
        for (int j = 0; j < M.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(M, j); it; ++it) {
                it.valueRef() *= delta(it.row()) * delta(j);
            }
        }

        // Accumulate
        cum.array() *= delta.array();
    }

    // Split into D_ (primal) and E_ (dual)
    D_ = cum.head(ny_);
    E_ = cum.tail(n_eq_);
    D_inv_ = D_.cwiseInverse();

    // Scale bounds: l_s = D^{-1} * l, u_s = D^{-1} * u
    lower_bounds_scaled_ = D_inv_.asDiagonal() * lower_bounds_;
    upper_bounds_scaled_ = D_inv_.asDiagonal() * upper_bounds_;

    // Scale RHS: d_s = E * d
    d_scaled_ = E_.asDiagonal() * d_;
}

// ---------------------------------------------------------------------------
// Build scaled KKT matrix with rho*I as plain diagonal
//
//   [D*H*D + rho*I    (E*C*D)^T ]
//   [E*C*D            -E*eps*E  ]
//
// The rho*I is a PLAIN identity — NOT conjugated by D^2.
// This is the key difference from symmetric KKT preconditioning.
// ---------------------------------------------------------------------------
void ADMMSolver::buildScaledKKT(double rho) const {
    const int dim = ny_ + n_eq_;
    const bool has_x0 = data_.x0.has_value();
    const int n_dyn   = data_.N * nx_;

    const int nnz_estimate =
        data_.N * (nx_ * nx_ + nu_ * nu_) + nx_ * nx_
        + ny_   // rho*I diagonal
        + 2 * data_.N * (nx_ * nx_ + nx_ * nu_ + nx_)
        + n_eq_
        + (has_x0 ? 2 * nx_ : 0);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz_estimate);

    // --- Top-left block: D*H*D + rho*I ---
    int idx = 0;
    for (int k = 0; k < data_.N; ++k) {
        for (int i = 0; i < nx_; ++i)
            for (int j = 0; j < nx_; ++j)
                triplets.emplace_back(idx + i, idx + j,
                                      D_[idx + i] * data_.Q(i, j) * D_[idx + j]);
        idx += nx_;

        for (int i = 0; i < nu_; ++i)
            for (int j = 0; j < nu_; ++j)
                triplets.emplace_back(idx + i, idx + j,
                                      D_[idx + i] * data_.R(i, j) * D_[idx + j]);
        idx += nu_;
    }
    for (int i = 0; i < nx_; ++i)
        for (int j = 0; j < nx_; ++j)
            triplets.emplace_back(idx + i, idx + j,
                                  D_[idx + i] * data_.P(i, j) * D_[idx + j]);

    // rho*I as PLAIN diagonal (not D^2 * rho)
    for (int i = 0; i < ny_; ++i)
        triplets.emplace_back(i, i, rho);

    // --- Scaled dynamics constraints: E*C*D and (E*C*D)^T ---
    const int stride = nx_ + nu_;

    for (int k = 0; k < data_.N; ++k) {
        const int row     = k * nx_;
        const int col_xk  = k * stride;
        const int col_uk  = k * stride + nx_;
        const int col_xk1 = (k + 1) * stride;

        for (int i = 0; i < nx_; ++i) {
            const double e_i = E_[row + i];
            for (int j = 0; j < nx_; ++j) {
                const double val = -data_.A(i, j) * e_i * D_[col_xk + j];
                triplets.emplace_back(ny_ + row + i, col_xk + j, val);
                triplets.emplace_back(col_xk + j, ny_ + row + i, val);
            }
            for (int j = 0; j < nu_; ++j) {
                const double val = -data_.B(i, j) * e_i * D_[col_uk + j];
                triplets.emplace_back(ny_ + row + i, col_uk + j, val);
                triplets.emplace_back(col_uk + j, ny_ + row + i, val);
            }
            {
                const double val = e_i * D_[col_xk1 + i];
                triplets.emplace_back(ny_ + row + i, col_xk1 + i, val);
                triplets.emplace_back(col_xk1 + i, ny_ + row + i, val);
            }
        }
    }

    // --- Scaled initial state constraint ---
    if (has_x0) {
        const int row0 = n_dyn;
        for (int i = 0; i < nx_; ++i) {
            const double val = E_[row0 + i] * D_[i];
            triplets.emplace_back(ny_ + row0 + i, i, val);
            triplets.emplace_back(i, ny_ + row0 + i, val);
        }
    }

    // --- Bottom-right block: E * (-eps*I) * E ---
    for (int i = 0; i < n_eq_; ++i)
        triplets.emplace_back(ny_ + i, ny_ + i, -data_.kkt_reg * E_[i] * E_[i]);

    kkt_matrix_.resize(dim, dim);
    kkt_matrix_.setFromTriplets(triplets.begin(), triplets.end());
    kkt_matrix_.makeCompressed();
}

// ---------------------------------------------------------------------------
// Efficient rho update: only modify diagonal, skip symbolic analysis
// ---------------------------------------------------------------------------
void ADMMSolver::refactorize(double new_rho) const {
    // In-place diagonal update: only the rho*I part changes
    double* vals = kkt_matrix_.valuePtr();
    const double delta = new_rho - current_rho_;
    for (int i = 0; i < ny_; ++i) {
        vals[diag_offsets_[i]] += delta;
    }
    current_rho_ = new_rho;

    // Numerical factorization only (reuses symbolic analysis from constructor)
    solver_.factorize(kkt_matrix_);
}

// ---------------------------------------------------------------------------
// ADMM iterations with warm-start and adaptive rho
// ---------------------------------------------------------------------------
ADMMResult ADMMSolver::solve(const WarmStart& warm) const {
    ADMMResult result;
    result.time_kkt_us = kkt_time_us_;

    auto solve_start = std::chrono::high_resolution_clock::now();

    if (use_riccati_) {
        // ===== Riccati path: operate in ORIGINAL space =====
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

        for (int iter = 0; iter < data_.max_iter; ++iter) {
            y = riccatiYUpdate(rho, z, lambda);

            Eigen::VectorXd y_hat = data_.alpha * y + (1.0 - data_.alpha) * z;

            z_prev = z;
            Eigen::VectorXd v = y_hat + lambda / rho;
            z = v.cwiseMax(lower_bounds_).cwiseMin(upper_bounds_);

            lambda.noalias() += rho * (y_hat - z);

            // Primal residual uses over-relaxed ŷ (OSQP convention)
            double primal_res = (y_hat - z).lpNorm<Eigen::Infinity>();
            double dual_res   = (rho * (z - z_prev)).lpNorm<Eigen::Infinity>();

            double pri_norm = std::max(y_hat.lpNorm<Eigen::Infinity>(),
                                       z.lpNorm<Eigen::Infinity>());
            double dua_norm = lambda.lpNorm<Eigen::Infinity>();

            double pri_tol = data_.eps_abs + data_.eps_rel * pri_norm;
            double dua_tol = data_.eps_abs + data_.eps_rel * dua_norm;

            result.iterations      = iter + 1;
            result.primal_residual = primal_res;
            result.dual_residual   = dual_res;

            if (primal_res <= pri_tol && dual_res <= dua_tol) {
                result.converged = true;
                result.final_rho = rho;
                break;
            }

            if (data_.adaptive_rho &&
                data_.adapt_interval > 0 &&
                ((iter + 1) % data_.adapt_interval == 0)) {
                double dual_safe = std::max(dual_res, 1e-10);
                double rho_estimate = std::sqrt(primal_res / dual_safe);
                double rho_new = rho * std::pow(rho_estimate, 0.3);
                rho_new = std::max(rho_new, data_.rho_min);
                rho_new = std::min(rho_new, data_.rho_max);

                if (rho_new > rho * data_.adapt_tolerance ||
                    rho_new < rho / data_.adapt_tolerance) {
                    lambda *= (rho_new / rho);
                    rho = rho_new;
                }
            }
        }

        auto solve_end = std::chrono::high_resolution_clock::now();
        result.time_solve_us =
            std::chrono::duration<double, std::micro>(solve_end - solve_start).count();

        // Compute pre-polish metrics
        if (result.converged) {
            result.pre_polish_primal_residual = result.primal_residual;
            result.pre_polish_dual_residual   = result.dual_residual;
            result.pre_polish_objective_cost  = computeObjectiveCost(y);
            result.pre_polish_max_bound_violation = computeMaxBoundViolation(y);
        }

        // Polishing (original space)
        if (result.converged && data_.polish) {
            polishSolution(y, z, lambda, result);
        }

        // Extract trajectory
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

        // Compute final metrics
        if (result.converged) {
            result.objective_cost      = computeObjectiveCost(y);
            result.max_bound_violation = computeMaxBoundViolation(y);
        }

        return result;
    }

    // ===== LDLT path with Ruiz scaling: operate in SCALED space =====
    Eigen::VectorXd y_s, z_s, z_prev_s, lambda_s;

    if (warm.y.size() == ny_ && warm.z.size() == ny_ && warm.lambda.size() == ny_) {
        // Convert warm-start from original to scaled space
        y_s      = D_inv_.asDiagonal() * warm.y;
        z_s      = D_inv_.asDiagonal() * warm.z;
        lambda_s = D_inv_.asDiagonal() * warm.lambda;
    } else {
        y_s      = Eigen::VectorXd::Zero(ny_);
        z_s      = Eigen::VectorXd::Zero(ny_);
        lambda_s = Eigen::VectorXd::Zero(ny_);
    }
    z_prev_s = z_s;

    double rho = data_.rho;

    Eigen::VectorXd rhs(ny_ + n_eq_);
    rhs.tail(n_eq_) = d_scaled_;

    for (int iter = 0; iter < data_.max_iter; ++iter) {
        // --- Step 1: y-update (scaled space) ---
        rhs.head(ny_) = rho * z_s - lambda_s;
        Eigen::VectorXd sol = solver_.solve(rhs);
        if (solver_.info() != Eigen::Success) {
            result.converged = false;
            result.iterations = iter;
            result.final_rho = rho;
            break;
        }
        y_s = sol.head(ny_);

        // Over-relaxed y (scaled space)
        Eigen::VectorXd y_hat_s = data_.alpha * y_s + (1.0 - data_.alpha) * z_s;

        // --- Step 2: z-update (scaled space, scaled bounds) ---
        z_prev_s = z_s;
        Eigen::VectorXd v_s = y_hat_s + lambda_s / rho;
        z_s = v_s.cwiseMax(lower_bounds_scaled_).cwiseMin(upper_bounds_scaled_);

        // --- Step 3: lambda-update (scaled space) ---
        lambda_s.noalias() += rho * (y_hat_s - z_s);

        // --- Convergence check in ORIGINAL space ---
        // Use over-relaxed ŷ for primal residual (OSQP convention):
        //   pri_res = ‖ŷ - z‖  (not ‖y - z‖)
        Eigen::VectorXd pri_diff = D_.asDiagonal() * (y_hat_s - z_s);
        Eigen::VectorXd dua_diff = D_.asDiagonal() * (z_s - z_prev_s);

        double primal_res = pri_diff.lpNorm<Eigen::Infinity>();
        double dual_res   = rho * dua_diff.lpNorm<Eigen::Infinity>();

        double pri_norm = std::max((D_.asDiagonal() * y_hat_s).lpNorm<Eigen::Infinity>(),
                                   (D_.asDiagonal() * z_s).lpNorm<Eigen::Infinity>());
        double dua_norm = (D_.asDiagonal() * lambda_s).lpNorm<Eigen::Infinity>();

        double pri_tol = data_.eps_abs + data_.eps_rel * pri_norm;
        double dua_tol = data_.eps_abs + data_.eps_rel * dua_norm;

        result.iterations      = iter + 1;
        result.primal_residual = primal_res;
        result.dual_residual   = dual_res;

        if (primal_res <= pri_tol && dual_res <= dua_tol) {
            result.converged = true;
            result.final_rho = rho;
            break;
        }

        // --- Adaptive rho (OSQP-style tempered update) ---
        if (data_.adaptive_rho &&
            data_.adapt_interval > 0 &&
            ((iter + 1) % data_.adapt_interval == 0)) {
            double dual_safe = std::max(dual_res, 1e-10);
            double rho_estimate = std::sqrt(primal_res / dual_safe);
            double rho_new = rho * std::pow(rho_estimate, 0.3);
            rho_new = std::max(rho_new, data_.rho_min);
            rho_new = std::min(rho_new, data_.rho_max);

            if (rho_new > rho * data_.adapt_tolerance ||
                rho_new < rho / data_.adapt_tolerance) {
                lambda_s *= (rho_new / rho);
                rho = rho_new;
                refactorize(rho);
            }
        }
    }

    auto solve_end = std::chrono::high_resolution_clock::now();
    result.time_solve_us =
        std::chrono::duration<double, std::micro>(solve_end - solve_start).count();

    // Unscale: y = D * y_s
    Eigen::VectorXd y = D_.asDiagonal() * y_s;

    // Compute pre-polish metrics
    if (result.converged) {
        result.pre_polish_primal_residual = result.primal_residual;
        result.pre_polish_dual_residual   = result.dual_residual;
        result.pre_polish_objective_cost  = computeObjectiveCost(y);
        result.pre_polish_max_bound_violation = computeMaxBoundViolation(y);
    }

    // Polishing (original space)
    if (result.converged && data_.polish) {
        Eigen::VectorXd z_orig     = D_.asDiagonal() * z_s;
        Eigen::VectorXd lambda_orig = D_.asDiagonal() * lambda_s;
        polishSolution(y, z_orig, lambda_orig, result);
    }

    // Extract trajectory from y (now in original space)
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

    // Compute final metrics
    if (result.converged) {
        result.objective_cost      = computeObjectiveCost(y);
        result.max_bound_violation = computeMaxBoundViolation(y);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Compute objective cost: 0.5 * sum(x_k^T Q x_k + u_k^T R u_k) + 0.5 * x_N^T P x_N
// ---------------------------------------------------------------------------
double ADMMSolver::computeObjectiveCost(const Eigen::VectorXd& y) const {
    const int stride = nx_ + nu_;
    double cost = 0.0;
    for (int k = 0; k < data_.N; ++k) {
        int base = k * stride;
        cost += 0.5 * y.segment(base, nx_).dot(data_.Q * y.segment(base, nx_));
        cost += 0.5 * y.segment(base + nx_, nu_).dot(data_.R * y.segment(base + nx_, nu_));
    }
    int term_base = data_.N * stride;
    cost += 0.5 * y.segment(term_base, nx_).dot(data_.P * y.segment(term_base, nx_));
    return cost;
}

// ---------------------------------------------------------------------------
// Compute max bound violation: max(0, max(|y_i - clip(y_i, l_i, u_i)|))
// ---------------------------------------------------------------------------
double ADMMSolver::computeMaxBoundViolation(const Eigen::VectorXd& y) const {
    double violation = 0.0;
    for (int i = 0; i < ny_; ++i) {
        double lo = lower_bounds_[i];
        double hi = upper_bounds_[i];
        if (y[i] < lo) violation = std::max(violation, lo - y[i]);
        else if (y[i] > hi) violation = std::max(violation, y[i] - hi);
    }
    return violation;
}

// ---------------------------------------------------------------------------
// Riccati y-update: extract linear terms, run backward + forward pass
// ---------------------------------------------------------------------------
Eigen::VectorXd ADMMSolver::riccatiYUpdate(
    double rho,
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& lambda) const {

    const int stride = nx_ + nu_;

    // Linear terms: q = -(rho*z - lambda)
    std::vector<Eigen::VectorXd> q_x(data_.N + 1);
    std::vector<Eigen::VectorXd> q_u(data_.N);

    for (int k = 0; k < data_.N; ++k) {
        int base = k * stride;
        q_x[k] = -(rho * z.segment(base, nx_) - lambda.segment(base, nx_));
        q_u[k] = -(rho * z.segment(base + nx_, nu_) - lambda.segment(base + nx_, nu_));
    }
    int term_base = data_.N * stride;
    q_x[data_.N] = -(rho * z.segment(term_base, nx_) - lambda.segment(term_base, nx_));

    // Cache gains (no-op if rho unchanged) + update linear terms
    riccati_->cacheGains(data_.A, data_.B, data_.Q, data_.R, data_.P, rho);
    riccati_->updateLinear(q_x, q_u);

    // Forward pass
    Eigen::VectorXd x0_vec;
    if (data_.x0.has_value()) {
        x0_vec = *data_.x0;
    }
    return riccati_->forward(x0_vec, data_.A, data_.B);
}

// ---------------------------------------------------------------------------
// OSQP-style polishing: identify active set, solve reduced equality-constrained QP
// ---------------------------------------------------------------------------
void ADMMSolver::polishSolution(
    Eigen::VectorXd& y,
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& lambda,
    ADMMResult& result) const
{
    auto t0 = std::chrono::high_resolution_clock::now();

    const double delta = data_.polish_delta;

    // --- Step 1: Active set identification (OSQP criterion) ---
    // Lower-active: z_i - l_i < -lambda_i (small gap, negative dual)
    // Upper-active: u_i - z_i < lambda_i  (small gap, positive dual)
    // Equality:     l_i == u_i
    std::vector<int> active_idx;
    std::vector<double> active_val;

    for (int i = 0; i < ny_; ++i) {
        bool is_equality = std::abs(upper_bounds_[i] - lower_bounds_[i]) < 1e-12;
        if (is_equality) {
            active_idx.push_back(i);
            active_val.push_back(lower_bounds_[i]);
        } else if (z[i] - lower_bounds_[i] < -lambda[i]) {
            active_idx.push_back(i);
            active_val.push_back(lower_bounds_[i]);
        } else if (upper_bounds_[i] - z[i] < lambda[i]) {
            active_idx.push_back(i);
            active_val.push_back(upper_bounds_[i]);
        }
    }

    int n_active = static_cast<int>(active_idx.size());
    if (n_active == 0) {
        // No active bounds → unconstrained optimum is feasible, nothing to polish
        return;
    }

    // --- Step 2: Build polishing KKT (original space, no Ruiz) ---
    //
    // [H + δI      C^T       S^T    ] [y  ]   [δ·y_admm]
    // [C           -εI       0      ] [ν  ] = [d       ]
    // [S            0       -δI     ] [μ  ]   [b_act   ]
    //
    // where S is n_active × ny selection matrix for active bound constraints,
    // and b_act are the corresponding bound values.

    const int dim = ny_ + n_eq_ + n_active;
    const bool has_x0 = data_.x0.has_value();
    const int n_dyn   = data_.N * nx_;
    const int stride  = nx_ + nu_;

    // Equality constraint RHS: d
    Eigen::VectorXd d(n_eq_);
    d.setZero();
    if (has_x0) d.tail(nx_) = *data_.x0;

    const int nnz_estimate =
        data_.N * (nx_ * nx_ + nu_ * nu_) + nx_ * nx_
        + ny_ + n_eq_ + n_active            // diagonal entries
        + 2 * data_.N * (nx_ * nx_ + nx_ * nu_ + nx_)
        + (has_x0 ? 2 * nx_ : 0)
        + 2 * n_active;                      // S and S^T

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz_estimate);

    // --- Top-left block: H + δI ---
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

    // δI diagonal
    for (int i = 0; i < ny_; ++i)
        triplets.emplace_back(i, i, delta);

    // --- Dynamics constraints C and C^T ---
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

    // Initial state constraint
    if (has_x0) {
        const int row0 = n_dyn;
        for (int i = 0; i < nx_; ++i) {
            triplets.emplace_back(ny_ + row0 + i, i, 1.0);
            triplets.emplace_back(i, ny_ + row0 + i, 1.0);
        }
    }

    // --- Bottom-right (dynamics): -εI ---
    for (int i = 0; i < n_eq_; ++i)
        triplets.emplace_back(ny_ + i, ny_ + i, -data_.kkt_reg);

    // --- Active bound constraints S and S^T ---
    const int s_row = ny_ + n_eq_;
    for (int i = 0; i < n_active; ++i) {
        int col = active_idx[i];
        triplets.emplace_back(s_row + i, col, 1.0);
        triplets.emplace_back(col, s_row + i, 1.0);
    }

    // --- Active constraint diagonal: -δI ---
    for (int i = 0; i < n_active; ++i)
        triplets.emplace_back(s_row + i, s_row + i, -delta);

    // --- Build and factorize ---
    Eigen::SparseMatrix<double> K(dim, dim);
    K.setFromTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> polish_solver;
    polish_solver.compute(K);
    if (polish_solver.info() != Eigen::Success) return;

    // --- RHS ---
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(dim);
    rhs.head(ny_) = delta * y;                // warm-start regularization
    rhs.segment(ny_, n_eq_) = d;              // dynamics RHS
    for (int i = 0; i < n_active; ++i)
        rhs(s_row + i) = active_val[i];       // bound values

    Eigen::VectorXd sol = polish_solver.solve(rhs);
    if (polish_solver.info() != Eigen::Success) return;

    // --- Step 3: Iterative refinement ---
    // Remove regularization error by refining against the true KKT:
    //   res = [0; d; b_act] - K_true * sol
    // Using: K_true = K_reg - diag(δI, -εI, -δI) + diag(0, 0, 0)
    //       (only (1,1) and (3,3) blocks differ by δ)
    // And:  K_reg * sol_0 = [δ*y_admm; d; b_act]
    //
    // After initial solve, for each refinement iteration:
    //   res = true_rhs - K_true * sol
    //   dsol = K_reg \ res
    //   sol += dsol
    for (int r = 0; r < data_.polish_refine_iter; ++r) {
        Eigen::VectorXd true_rhs = Eigen::VectorXd::Zero(dim);
        true_rhs.segment(ny_, n_eq_) = d;
        for (int i = 0; i < n_active; ++i)
            true_rhs(s_row + i) = active_val[i];

        // K_true * sol: compute residual using K_reg product
        // res = true_rhs - K_true * sol
        //     = true_rhs - (K_reg - D_reg) * sol    where D_reg = diag(δ,0,...,0, -ε, ..., -δ)
        //     = true_rhs - K_reg*sol + D_reg*sol
        //
        // Since K_reg * sol = rhs was the initial solve, but sol has been updated:
        // We compute K_reg * sol explicitly (matrix-vector product)
        Eigen::VectorXd Kreg_sol = K * sol;
        // D_reg * sol: only the extra diagonal terms
        Eigen::VectorXd Dreg_sol = Eigen::VectorXd::Zero(dim);
        for (int i = 0; i < ny_; ++i)
            Dreg_sol(i) = delta * sol(i);
        for (int i = 0; i < n_eq_; ++i)
            Dreg_sol(ny_ + i) = -data_.kkt_reg * sol(ny_ + i);
        for (int i = 0; i < n_active; ++i)
            Dreg_sol(s_row + i) = -delta * sol(s_row + i);

        Eigen::VectorXd res = true_rhs - Kreg_sol + Dreg_sol;

        Eigen::VectorXd dsol = polish_solver.solve(res);
        if (polish_solver.info() != Eigen::Success) break;
        sol += dsol;
    }

    // --- Step 4: Extract and validate ---
    Eigen::VectorXd y_pol = sol.head(ny_);

    // Check feasibility (with tolerance)
    const double feas_tol = 1e-6;
    for (int i = 0; i < ny_; ++i) {
        if (y_pol[i] < lower_bounds_[i] - feas_tol ||
            y_pol[i] > upper_bounds_[i] + feas_tol) {
            return;  // infeasible, keep ADMM solution
        }
    }

    // Clip to ensure strict feasibility
    y_pol = y_pol.cwiseMax(lower_bounds_).cwiseMin(upper_bounds_);

    // Accept polished solution
    y = y_pol;
    result.polished = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    result.time_polish_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
}

}  // namespace admm
