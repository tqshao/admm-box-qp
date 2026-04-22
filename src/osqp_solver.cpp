#include "admm/osqp_solver.h"

#include <osqp.h>
#include <chrono>
#include <stdexcept>
#include <algorithm>

namespace admm {

// ---------------------------------------------------------------------------
// Constructor: build CSC matrices and call osqp_setup()
// ---------------------------------------------------------------------------
OsqpSolver::OsqpSolver(const ProblemData& data) : data_(data) {
    nx_ = data_.nx();
    nu_ = data_.nu();
    ny_ = data_.ny();

    const bool has_x0 = data_.x0.has_value();
    const int n_dyn   = data_.N * nx_;
    n_eq_             = n_dyn + (has_x0 ? nx_ : 0);
    n_constr_         = n_eq_ + ny_;

    auto t0 = std::chrono::high_resolution_clock::now();

    // --- Build cost matrix P = H (upper triangle, CSC) ---
    {
        std::vector<c_float> Px;
        std::vector<c_int>   Pi, Pp;
        Px.reserve(ny_);
        Pi.reserve(ny_);
        Pp.resize(ny_ + 1);

        int idx = 0;
        Pp[0] = 0;
        auto addDiag = [&](const Eigen::MatrixXd& M, int dim) {
            for (int j = 0; j < dim; ++j, ++idx) {
                Px.push_back(M(j, j));
                Pi.push_back(idx);
                Pp[idx + 1] = Pp[idx] + 1;
            }
        };
        for (int k = 0; k < data_.N; ++k) {
            addDiag(data_.Q, nx_);
            addDiag(data_.R, nu_);
        }
        addDiag(data_.P, nx_);

        // q = 0
        std::vector<c_float> q(ny_, 0.0);

        // --- Build constraint matrix A = [C; I] and bounds ---
        using Entry = std::pair<c_int, c_float>;
        std::vector<std::vector<Entry>> columns(ny_);

        const int stride = nx_ + nu_;

        // Dynamics: x_{k+1} - A x_k - B u_k = 0
        for (int k = 0; k < data_.N; ++k) {
            const int row_base = k * nx_;
            for (int j = 0; j < nx_; ++j) {
                int col = k * stride + j;
                for (int i = 0; i < nx_; ++i)
                    columns[col].push_back({row_base + i, -data_.A(i, j)});
            }
            for (int j = 0; j < nu_; ++j) {
                int col = k * stride + nx_ + j;
                for (int i = 0; i < nx_; ++i)
                    columns[col].push_back({row_base + i, -data_.B(i, j)});
            }
            for (int j = 0; j < nx_; ++j) {
                int col = (k + 1) * stride + j;
                columns[col].push_back({row_base + j, 1.0});
            }
        }

        if (has_x0) {
            for (int j = 0; j < nx_; ++j)
                columns[j].push_back({n_dyn + j, 1.0});
        }

        for (int j = 0; j < ny_; ++j)
            columns[j].push_back({n_eq_ + j, 1.0});

        // Sort and flatten
        std::vector<c_float> Ax;
        std::vector<c_int>   Ai, Ap(ny_ + 1);
        Ap[0] = 0;
        for (int j = 0; j < ny_; ++j) {
            std::sort(columns[j].begin(), columns[j].end());
            for (const auto& [row, val] : columns[j]) {
                Ax.push_back(val);
                Ai.push_back(row);
            }
            Ap[j + 1] = static_cast<c_int>(Ax.size());
        }

        // Bounds
        std::vector<c_float> l(n_constr_), u(n_constr_);
        for (int i = 0; i < n_dyn; ++i) {
            l[i] = 0.0;  u[i] = 0.0;
        }
        if (has_x0) {
            for (int i = 0; i < nx_; ++i) {
                l[n_dyn + i] = (*data_.x0)(i);
                u[n_dyn + i] = (*data_.x0)(i);
            }
        }

        Eigen::VectorXd lo(ny_), hi(ny_);
        int bidx = 0;
        for (int k = 0; k < data_.N; ++k) {
            lo.segment(bidx, nx_) = data_.x_min;
            hi.segment(bidx, nx_) = data_.x_max;
            bidx += nx_;
            lo.segment(bidx, nu_) = data_.u_min;
            hi.segment(bidx, nu_) = data_.u_max;
            bidx += nu_;
        }
        lo.segment(bidx, nx_) = data_.x_min;
        hi.segment(bidx, nx_) = data_.x_max;

        if (data_.custom_lower_bounds.has_value())
            lo = *data_.custom_lower_bounds;
        if (data_.custom_upper_bounds.has_value())
            hi = *data_.custom_upper_bounds;

        for (int i = 0; i < ny_; ++i) {
            l[n_eq_ + i] = lo(i);
            u[n_eq_ + i] = hi(i);
        }

        // --- OSQP data ---
        OSQPData* osqp_data = (OSQPData*)c_malloc(sizeof(OSQPData));
        osqp_data->n = ny_;
        osqp_data->m = n_constr_;
        osqp_data->P = csc_matrix(ny_, ny_, static_cast<c_int>(Px.size()),
                                  Px.data(), Pi.data(), Pp.data());
        osqp_data->q = q.data();
        osqp_data->A = csc_matrix(n_constr_, ny_, static_cast<c_int>(Ax.size()),
                                  Ax.data(), Ai.data(), Ap.data());
        osqp_data->l = l.data();
        osqp_data->u = u.data();

        OSQPSettings* settings = (OSQPSettings*)c_malloc(sizeof(OSQPSettings));
        osqp_set_default_settings(settings);
        settings->verbose      = 0;
        settings->max_iter     = data_.max_iter;
        settings->eps_abs      = data_.eps_abs;
        settings->eps_rel      = data_.eps_rel;
        settings->polish       = 1;
        settings->adaptive_rho = 1;
        settings->warm_start   = 0;

        OSQPWorkspace* work = nullptr;
        c_int retval = osqp_setup(&work, osqp_data, settings);
        osqp_workspace_ = static_cast<void*>(work);

        if (osqp_data->A) c_free(osqp_data->A);
        if (osqp_data->P) c_free(osqp_data->P);
        c_free(osqp_data);
        c_free(settings);

        if (retval != 0) {
            throw std::runtime_error("OsqpSolver: osqp_setup failed with code " +
                                     std::to_string(retval));
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    setup_time_us_ = std::chrono::duration<double, std::micro>(t1 - t0).count();
}

OsqpSolver::~OsqpSolver() {
    if (osqp_workspace_) {
        osqp_cleanup(static_cast<OSQPWorkspace*>(osqp_workspace_));
    }
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------
OsqpResult OsqpSolver::solve() const {
    OsqpResult result;
    result.time_setup_us = setup_time_us_;

    OSQPWorkspace* work = static_cast<OSQPWorkspace*>(osqp_workspace_);

    auto t0 = std::chrono::high_resolution_clock::now();
    c_int retval = osqp_solve(work);
    auto t1 = std::chrono::high_resolution_clock::now();

    result.time_solve_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    result.converged = (retval == 0 &&
                        (work->info->status_val == OSQP_SOLVED ||
                         work->info->status_val == OSQP_SOLVED_INACCURATE));
    result.iterations = static_cast<int>(work->info->iter);
    result.primal_residual = work->info->pri_res;
    result.dual_residual = work->info->dua_res;

    if (retval == 0 || work->solution) {
        const c_float* sol = work->solution->x;
        result.x.resize(data_.N + 1);
        result.u.resize(data_.N);
        int idx = 0;
        for (int k = 0; k < data_.N; ++k) {
            result.x[k] = Eigen::VectorXd(nx_);
            for (int i = 0; i < nx_; ++i)
                result.x[k](i) = sol[idx++];
            result.u[k] = Eigen::VectorXd(nu_);
            for (int i = 0; i < nu_; ++i)
                result.u[k](i) = sol[idx++];
        }
        result.x[data_.N] = Eigen::VectorXd(nx_);
        for (int i = 0; i < nx_; ++i)
            result.x[data_.N](i) = sol[idx++];
    }

    return result;
}

}  // namespace admm
