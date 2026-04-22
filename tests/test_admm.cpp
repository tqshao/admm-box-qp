#include <gtest/gtest.h>

#include "admm/admm_solver.h"

using namespace admm;

// ============================================================================
// Helpers
// ============================================================================

// Build a double-integrator problem:
//   state  x = [position; velocity],  control u = [acceleration]
//   x_{k+1} = A x_k + B u_k
//
ProblemData makeDoubleIntegrator(
    int N, double dt,
    const Eigen::VectorXd& x0,
    const Eigen::VectorXd& x_min, const Eigen::VectorXd& x_max,
    const Eigen::VectorXd& u_min, const Eigen::VectorXd& u_max,
    double rho = 1.0, double alpha = 1.6) {
    ProblemData data;

    data.A = Eigen::MatrixXd::Identity(2, 2);
    data.A(0, 1) = dt;

    data.B = Eigen::MatrixXd::Zero(2, 1);
    data.B(0, 0) = 0.5 * dt * dt;
    data.B(1, 0) = dt;

    data.Q = Eigen::MatrixXd::Identity(2, 2);
    data.R = Eigen::MatrixXd::Identity(1, 1);
    data.P = 10.0 * Eigen::MatrixXd::Identity(2, 2);

    data.x_min = x_min;
    data.x_max = x_max;
    data.u_min = u_min;
    data.u_max = u_max;

    data.N        = N;
    data.rho      = rho;
    data.alpha    = alpha;
    data.eps_abs  = 1e-3;
    data.eps_rel = 1e-3;
    data.max_iter = 2000;
    data.x0       = x0;

    return data;
}

// ============================================================================
// Test 1: Unconstrained problem (very wide bounds) should converge
// ============================================================================
TEST(ADMMSolverTest, UnconstrainedConvergence) {
    const int N = 20;
    const double dt = 0.1;

    Eigen::VectorXd x0(2);
    x0 << 1.0, 0.0;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -1e6, -1e6;
    x_max <<  1e6,  1e6;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -1e6;
    u_max <<  1e6;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max);
    ADMMSolver solver(data);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged);
    EXPECT_LT(result.iterations, 1500);
    EXPECT_LT(result.primal_residual, data.eps_abs);
    EXPECT_LT(result.dual_residual, data.eps_rel);

    // Initial state must match x0 (enforced as equality constraint)
    EXPECT_NEAR(result.x[0](0), 1.0, 1e-4);
    EXPECT_NEAR(result.x[0](1), 0.0, 1e-4);
}

// ============================================================================
// Test 2: Input constraints are satisfied
// ============================================================================
TEST(ADMMSolverTest, InputBoundsSatisfied) {
    const int N = 20;
    const double dt = 0.1;

    Eigen::VectorXd x0(2);
    x0 << 1.0, 0.0;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -10.0, -10.0;
    x_max <<  10.0,  10.0;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -0.5;
    u_max <<  0.5;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max);
    ADMMSolver solver(data);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged);

    // The primal variable y is extracted; it satisfies dynamics exactly but
    // box constraints only up to the primal residual tolerance (eps_abs).
    for (int k = 0; k < N; ++k) {
        EXPECT_GE(result.u[k](0), u_min(0) - data.eps_abs)
            << "u[" << k << "] = " << result.u[k](0) << " < u_min = " << u_min(0);
        EXPECT_LE(result.u[k](0), u_max(0) + data.eps_abs)
            << "u[" << k << "] = " << result.u[k](0) << " > u_max = " << u_max(0);
    }
}

// ============================================================================
// Test 3: State and input constraints are simultaneously satisfied
// ============================================================================
TEST(ADMMSolverTest, StateAndInputBoundsSatisfied) {
    const int N = 20;
    const double dt = 0.1;

    Eigen::VectorXd x0(2);
    x0 << 2.0, 1.0;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -3.0, -3.0;
    x_max <<  3.0,  3.0;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -1.0;
    u_max <<  1.0;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max, 10.0);
    ADMMSolver solver(data);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged);

    // Check state bounds
    for (int k = 0; k <= N; ++k) {
        for (int i = 0; i < 2; ++i) {
            EXPECT_GE(result.x[k](i), x_min(i) - 1e-2)
                << "x[" << k << "](" << i << ") = " << result.x[k](i);
            EXPECT_LE(result.x[k](i), x_max(i) + 1e-2)
                << "x[" << k << "](" << i << ") = " << result.x[k](i);
        }
    }

    // Check input bounds (allow small violation from combined tolerance)
    for (int k = 0; k < N; ++k) {
        EXPECT_GE(result.u[k](0), u_min(0) - 1e-3);
        EXPECT_LE(result.u[k](0), u_max(0) + 1e-3);
    }
}

// ============================================================================
// Test 4: Dynamics consistency  x_{k+1} = A x_k + B u_k
// ============================================================================
TEST(ADMMSolverTest, DynamicsConsistency) {
    const int N = 15;
    const double dt = 0.1;

    Eigen::VectorXd x0(2);
    x0 << 1.0, 0.5;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -10.0, -10.0;
    x_max <<  10.0,  10.0;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -1.0;
    u_max <<  1.0;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max);
    ADMMSolver solver(data);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged);

    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd x_next = data.A * result.x[k] + data.B * result.u[k];
        EXPECT_NEAR(x_next(0), result.x[k + 1](0), 1e-8)
            << "Dynamics violated at step " << k << " (position)";
        EXPECT_NEAR(x_next(1), result.x[k + 1](1), 1e-8)
            << "Dynamics violated at step " << k << " (velocity)";
    }
}

// ============================================================================
// Test 5: Simple 1D system  (nx=1, nu=1, A=1, B=1)
// ============================================================================
TEST(ADMMSolverTest, Simple1DSystem) {
    ProblemData data;
    data.A = Eigen::MatrixXd::Identity(1, 1);
    data.B = Eigen::MatrixXd::Identity(1, 1);
    data.Q = Eigen::MatrixXd::Identity(1, 1);
    data.R = 0.1 * Eigen::MatrixXd::Identity(1, 1);
    data.P = 10.0 * Eigen::MatrixXd::Identity(1, 1);

    data.x_min = Eigen::VectorXd::Constant(1, -10.0);
    data.x_max = Eigen::VectorXd::Constant(1,  10.0);
    data.u_min = Eigen::VectorXd::Constant(1, -1.0);
    data.u_max = Eigen::VectorXd::Constant(1,  1.0);

    data.N        = 10;
    data.rho      = 1.0;
    data.alpha    = 1.6;
    data.eps_abs  = 1e-3;
    data.eps_rel = 1e-3;
    data.max_iter = 2000;

    Eigen::VectorXd x0(1);
    x0 << 5.0;
    data.x0 = x0;

    ADMMSolver solver(data);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged);

    // Initial state
    EXPECT_NEAR(result.x[0](0), 5.0, 1e-3);

    // The LQR controller should drive the state toward zero
    EXPECT_NEAR(result.x[data.N](0), 0.0, 0.5);

    // Dynamics: x_{k+1} = x_k + u_k
    for (int k = 0; k < data.N; ++k) {
        double x_next = result.x[k](0) + result.u[k](0);
        EXPECT_NEAR(x_next, result.x[k + 1](0), 1e-8)
            << "Dynamics violated at step " << k;
    }
}

// ============================================================================
// Test 6: Active state constraints — tight position bound with high velocity
// ============================================================================
TEST(ADMMSolverTest, ActiveStateConstraints) {
    const int N = 20;
    const double dt = 0.1;

    // Start at origin with high velocity → position will grow, hitting the
    // tight position bound unless the controller acts.
    Eigen::VectorXd x0(2);
    x0 << 0.0, 2.0;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -0.5, -5.0;
    x_max <<  0.5,  5.0;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -5.0;
    u_max <<  5.0;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max, 10.0);
    data.max_iter = 3000;
    ADMMSolver solver(data);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged);

    // Verify all states are within bounds (with tolerance for ADMM accuracy)
    for (int k = 0; k <= N; ++k) {
        EXPECT_GE(result.x[k](0), x_min(0) - 1e-2)
            << "State position lower bound violated at k=" << k;
        EXPECT_LE(result.x[k](0), x_max(0) + 1e-2)
            << "State position upper bound violated at k=" << k;
        EXPECT_GE(result.x[k](1), x_min(1) - 1e-2)
            << "State velocity lower bound violated at k=" << k;
        EXPECT_LE(result.x[k](1), x_max(1) + 1e-2)
            << "State velocity upper bound violated at k=" << k;
    }
}

// ============================================================================
// Test 7: Residuals decrease (monotonic improvement)
// ============================================================================
TEST(ADMMSolverTest, ResidualsBelowThreshold) {
    const int N = 15;
    const double dt = 0.1;

    Eigen::VectorXd x0(2);
    x0 << 2.0, 1.0;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -5.0, -5.0;
    x_max <<  5.0,  5.0;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -1.0;
    u_max <<  1.0;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max);
    data.max_iter = 5000;

    ADMMSolver solver(data);
    auto result = solver.solve();

    // At the very least residuals should be finite and small
    // Use the same combined tolerance as the solver: eps_abs + eps_rel * norm
    double tol = data.eps_abs + data.eps_rel * 10.0;  // conservative relative factor
    EXPECT_TRUE(std::isfinite(result.primal_residual));
    EXPECT_TRUE(std::isfinite(result.dual_residual));
    EXPECT_LT(result.primal_residual, tol);
    EXPECT_LT(result.dual_residual, tol);
}

// ============================================================================
// Test 8: Riccati path matches KKT path
// ============================================================================
TEST(ADMMSolverTest, RiccatiMatchesKKT) {
    const int N = 20;
    const double dt = 0.1;

    Eigen::VectorXd x0(2);
    x0 << 2.0, 1.0;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -3.0, -3.0;
    x_max <<  3.0,  3.0;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -1.0;
    u_max <<  1.0;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max, 10.0);

    // KKT path
    ADMMSolver solver_kkt(data);
    auto result_kkt = solver_kkt.solve();

    // Riccati path
    data.use_riccati = true;
    ADMMSolver solver_riccati(data);
    auto result_riccati = solver_riccati.solve();

    EXPECT_TRUE(result_kkt.converged);
    EXPECT_TRUE(result_riccati.converged);

    // Ruiz scaling changes numerical conditioning of the LDLT path,
    // so both paths converge within ADMM tolerance but to slightly
    // different numerical points.  Tolerance is consistent with
    // eps_abs + eps_rel * norm for the ADMM convergence criterion.
    for (int k = 0; k <= N; ++k) {
        EXPECT_NEAR(result_kkt.x[k](0), result_riccati.x[k](0), 5e-3)
            << "Position mismatch at step " << k;
        EXPECT_NEAR(result_kkt.x[k](1), result_riccati.x[k](1), 5e-3)
            << "Velocity mismatch at step " << k;
    }
    for (int k = 0; k < N; ++k) {
        EXPECT_NEAR(result_kkt.u[k](0), result_riccati.u[k](0), 5e-3)
            << "Input mismatch at step " << k;
    }
}

// ============================================================================
// Test 9: Riccati with adaptive rho
// ============================================================================
TEST(ADMMSolverTest, RiccatiWithAdaptiveRho) {
    const int N = 15;
    const double dt = 0.1;

    Eigen::VectorXd x0(2);
    x0 << 1.0, 0.5;

    Eigen::VectorXd x_min(2), x_max(2);
    x_min << -10.0, -10.0;
    x_max <<  10.0,  10.0;

    Eigen::VectorXd u_min(1), u_max(1);
    u_min << -1.0;
    u_max <<  1.0;

    auto data = makeDoubleIntegrator(N, dt, x0, x_min, x_max, u_min, u_max);
    data.use_riccati = true;
    data.adaptive_rho = true;

    ADMMSolver solver(data);
    auto result = solver.solve();

    EXPECT_TRUE(result.converged);

    // Dynamics consistency
    for (int k = 0; k < N; ++k) {
        Eigen::VectorXd x_next = data.A * result.x[k] + data.B * result.u[k];
        EXPECT_NEAR(x_next(0), result.x[k + 1](0), 1e-8);
        EXPECT_NEAR(x_next(1), result.x[k + 1](1), 1e-8);
    }
}
