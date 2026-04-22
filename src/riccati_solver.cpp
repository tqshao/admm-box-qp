#include "admm/riccati_solver.h"

namespace admm {

RiccatiSolver::RiccatiSolver(int nx, int nu, int N)
    : nx_(nx), nu_(nu), N_(N) {
    S_.resize(N + 1);
    K_.resize(N);
    Sigma_inv_.resize(N);
    Sigma_inv_Bt_.resize(N);
    AtSnextB_.resize(N);
    s_.resize(N + 1);
    kff_.resize(N);
    for (int k = 0; k < N; ++k) {
        S_[k].resize(nx, nx);
        K_[k].resize(nu, nx);
        Sigma_inv_[k].resize(nu, nu);
        Sigma_inv_Bt_[k].resize(nu, nx);
        AtSnextB_[k].resize(nx, nu);
        kff_[k].resize(nu);
    }
    S_[N].resize(nx, nx);
}

// ---------------------------------------------------------------------------
// cacheGains: heavy backward pass — S_k, K_k, Σ^{-1}, Σ^{-1}B^T, A^T S_{k+1} B
// Called once (or when ρ changes).  Everything here only depends on (A,B,Q,R,P,ρ).
// ---------------------------------------------------------------------------
void RiccatiSolver::cacheGains(
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P, double rho) {

    if (rho == cached_rho_) return;
    cached_rho_ = rho;
    cached_At_ = A.transpose();

    Eigen::MatrixXd Qbar = Q + rho * Eigen::MatrixXd::Identity(nx_, nx_);
    Eigen::MatrixXd Rbar = R + rho * Eigen::MatrixXd::Identity(nu_, nu_);
    Eigen::MatrixXd Bt = B.transpose();

    S_[N_] = P + rho * Eigen::MatrixXd::Identity(nx_, nx_);

    for (int k = N_ - 1; k >= 0; --k) {
        Eigen::MatrixXd BtS = Bt * S_[k + 1];              // nu × nx
        Eigen::MatrixXd Sigma = Rbar + BtS * B;             // nu × nu

        Sigma_inv_[k] = Sigma.inverse();                    // nu × nu
        Sigma_inv_Bt_[k] = Sigma_inv_[k] * Bt;              // nu × nx
        K_[k] = Sigma_inv_Bt_[k] * S_[k + 1] * A;          // nu × nx
        AtSnextB_[k] = cached_At_ * S_[k + 1] * B;          // nx × nu

        S_[k] = Qbar + cached_At_ * S_[k + 1] * A
                - K_[k].transpose() * Sigma * K_[k];
    }
}

// ---------------------------------------------------------------------------
// updateLinear: light per-iteration backward pass — only s_k and kff_k
//
// Per step this does:
//   kff_k = Σ^{-1}B^T s_{k+1}  +  Σ^{-1} q_u[k]    (mat-vec + mat-vec)
//   s_k   = q_x[k] + A^T s_{k+1} - (A^T S_{k+1}B) kff_k  (mat-vec + mat-vec)
//
// No matrix-matrix multiplications — O(N * (nx^2 + nx*nu)) per call.
// ---------------------------------------------------------------------------
void RiccatiSolver::updateLinear(
    const std::vector<Eigen::VectorXd>& q_x,
    const std::vector<Eigen::VectorXd>& q_u) {

    s_[N_] = q_x[N_];

    for (int k = N_ - 1; k >= 0; --k) {
        kff_[k] = Sigma_inv_Bt_[k] * s_[k + 1]             // nu×nx * nx×1
                  + Sigma_inv_[k] * q_u[k];                 // nu×nu * nu×1

        s_[k] = q_x[k]
                + cached_At_ * s_[k + 1]                    // nx×nx * nx×1
                - AtSnextB_[k] * kff_[k];                   // nx×nu * nu×1
    }
}

// ---------------------------------------------------------------------------
// Forward pass: roll out optimal trajectory from x0
// ---------------------------------------------------------------------------
Eigen::VectorXd RiccatiSolver::forward(
    const Eigen::VectorXd& x0,
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B) const {

    Eigen::VectorXd y((N_ + 1) * nx_ + N_ * nu_);

    Eigen::VectorXd x(nx_);
    if (x0.size() == nx_) {
        x = x0;
    } else {
        x = -S_[0].ldlt().solve(s_[0]);
    }

    int idx = 0;
    y.segment(idx, nx_) = x;
    idx += nx_;

    for (int k = 0; k < N_; ++k) {
        Eigen::VectorXd u = -K_[k] * x - kff_[k];
        y.segment(idx, nu_) = u;
        idx += nu_;

        x = A * x + B * u;
        y.segment(idx, nx_) = x;
        idx += nx_;
    }

    return y;
}

}  // namespace admm
