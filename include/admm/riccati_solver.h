#pragma once

#include <Eigen/Dense>
#include <vector>

namespace admm {

// Riccati backward-forward solver for the ADMM y-update subproblem.
//
// The computation is split into:
//   1. cacheGains() — heavy work: S_k, K_k, Σ^{-1}, Σ^{-1}B^T, A^T S_{k+1} B
//      Only depends on (A, B, Q, R, P, ρ).  No-op when ρ unchanged.
//   2. updateLinear() — light work: s_k, kff_k from linear terms (q_x, q_u)
//      Called every ADMM iteration.  Only matrix-vector products (no mat-mat).
//   3. forward() — rollout from x0 using cached K and per-iteration kff.
class RiccatiSolver {
public:
    RiccatiSolver(int nx, int nu, int N);

    // Cache gain matrices.  No-op if rho unchanged since last call.
    void cacheGains(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                    const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R,
                    const Eigen::MatrixXd& P, double rho);

    // Per-iteration: compute s_k and kff_k from linear terms.
    void updateLinear(const std::vector<Eigen::VectorXd>& q_x,
                      const std::vector<Eigen::VectorXd>& q_u);

    // Roll out optimal trajectory.  Returns stacked y = [x0;u0;x1;u1;...;xN].
    Eigen::VectorXd forward(const Eigen::VectorXd& x0,
                            const Eigen::MatrixXd& A,
                            const Eigen::MatrixXd& B) const;

private:
    int nx_, nu_, N_;

    // --- Cached gains (constant when ρ fixed) ---
    std::vector<Eigen::MatrixXd> S_;             // [N+1] nx×nx  cost-to-go Hessian
    std::vector<Eigen::MatrixXd> K_;             // [N]   nu×nx  feedback gain
    std::vector<Eigen::MatrixXd> Sigma_inv_;     // [N]   nu×nu
    std::vector<Eigen::MatrixXd> Sigma_inv_Bt_;  // [N]   nu×nx  Σ^{-1} B^T
    std::vector<Eigen::MatrixXd> AtSnextB_;      // [N]   nx×nu  A^T S_{k+1} B

    // --- Per-iteration quantities ---
    std::vector<Eigen::VectorXd> s_;             // [N+1] nx×1
    std::vector<Eigen::VectorXd> kff_;           // [N]   nu×1

    // Cached A^T (for s_k recurrence in updateLinear)
    Eigen::MatrixXd cached_At_;
    double cached_rho_ = -1.0;
};

}  // namespace admm
