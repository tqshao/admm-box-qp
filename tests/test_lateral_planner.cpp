#include <gtest/gtest.h>

#include "admm/lateral_planner.h"

using namespace admm;

// ============================================================================
// Helper: build a planner with default config but adjustable parameters
// ============================================================================
LateralPlanner makePlanner(double lane_half_width = 1.75,
                           double max_lat_vel    = 1.0,
                           double max_lat_acc    = 3.0,
                           double max_lat_jerk   = 10.0,
                           int    N              = 30) {
    PlannerConfig pc;
    pc.lane_half_width = lane_half_width;
    pc.max_lat_vel     = max_lat_vel;
    pc.max_lat_acc     = max_lat_acc;
    pc.max_lat_jerk    = max_lat_jerk;
    pc.N               = N;
    pc.v_x             = 20.0;
    pc.dt              = 0.1;

    SolverConfig sc;
    sc.rho = 10.0; sc.alpha = 1.6;
    sc.eps_pri = 1e-3; sc.eps_dual = 1e-3;
    sc.max_iter = 2000;

    return LateralPlanner(pc, sc);
}

// ============================================================================
// Test 1: Start at lane center → stays near center
// ============================================================================
TEST(LateralPlannerTest, LaneKeepingFromCenter) {
    auto planner = makePlanner();
    const auto& cfg = planner.plannerConfig();

    Eigen::VectorXd x0(3);
    x0 << 0.0, 0.0, 0.0;

    auto result = planner.plan(x0);

    EXPECT_TRUE(result.converged);

    // Initial state must match
    EXPECT_NEAR(result.x[0](0), 0.0, 1e-6);
    EXPECT_NEAR(result.x[0](1), 0.0, 1e-6);
    EXPECT_NEAR(result.x[0](2), 0.0, 1e-6);

    // Trajectory should stay close to center
    for (int k = 0; k <= cfg.N; ++k) {
        EXPECT_NEAR(result.x[k](0), 0.0, 0.1)
            << "Lateral offset at step " << k << ": " << result.x[k](0);
    }
}

// ============================================================================
// Test 2: Start at lateral offset → returns to center within bounds
// ============================================================================
TEST(LateralPlannerTest, LaneKeepingFromOffset) {
    auto planner = makePlanner();
    const auto& cfg = planner.plannerConfig();

    Eigen::VectorXd x0(3);
    x0 << 0.8, 0.0, 0.0;  // 0.8m offset, zero lateral velocity/acceleration

    auto result = planner.plan(x0);

    EXPECT_TRUE(result.converged);

    // Initial state
    EXPECT_NEAR(result.x[0](0), 0.8, 1e-4);

    // Terminal state should be closer to center than initial
    EXPECT_LT(std::abs(result.x[cfg.N](0)), std::abs(x0(0)));

    // All states within lane boundaries
    const double hw = cfg.lane_half_width;
    const double tol = planner.solverConfig().eps_pri;
    for (int k = 0; k <= cfg.N; ++k) {
        EXPECT_GE(result.x[k](0), -hw - tol)
            << "Left boundary violated at step " << k;
        EXPECT_LE(result.x[k](0), hw + tol)
            << "Right boundary violated at step " << k;
    }
}

// ============================================================================
// Test 3: Start near boundary → never crosses it
// ============================================================================
TEST(LateralPlannerTest, LaneBoundaryRespected) {
    auto planner = makePlanner(/*lane_half_width=*/1.5);
    const auto& cfg = planner.plannerConfig();
    const auto& sc  = planner.solverConfig();
    const double hw  = cfg.lane_half_width;
    const double tol = sc.eps_pri;

    Eigen::VectorXd x0(3);
    x0 << 1.3, 0.1, 0.0;  // very close to right boundary, drifting outward

    auto result = planner.plan(x0);

    EXPECT_TRUE(result.converged);

    for (int k = 0; k <= cfg.N; ++k) {
        EXPECT_GE(result.x[k](0), -hw - tol)
            << "Left boundary violated at step " << k
            << ": y=" << result.x[k](0);
        EXPECT_LE(result.x[k](0), hw + tol)
            << "Right boundary violated at step " << k
            << ": y=" << result.x[k](0);
        EXPECT_GE(result.x[k](1), -cfg.max_lat_vel - tol)
            << "Velocity lower bound violated at step " << k;
        EXPECT_LE(result.x[k](1), cfg.max_lat_vel + tol)
            << "Velocity upper bound violated at step " << k;
        EXPECT_GE(result.x[k](2), -cfg.max_lat_acc - tol)
            << "Acceleration lower bound violated at step " << k;
        EXPECT_LE(result.x[k](2), cfg.max_lat_acc + tol)
            << "Acceleration upper bound violated at step " << k;
    }

    for (int k = 0; k < cfg.N; ++k) {
        EXPECT_GE(result.u[k](0), -cfg.max_lat_jerk - tol)
            << "Jerk lower bound violated at step " << k;
        EXPECT_LE(result.u[k](0), cfg.max_lat_jerk + tol)
            << "Jerk upper bound violated at step " << k;
    }
}

// ============================================================================
// Test 4: Dynamics consistency  x_{k+1} = A x_k + B u_k
//         Triple integrator: state = [y, vy, ay], input = [jerk]
// ============================================================================
TEST(LateralPlannerTest, DynamicsConsistency) {
    auto planner = makePlanner();
    const auto& cfg = planner.plannerConfig();
    const double dt = cfg.dt;

    Eigen::VectorXd x0(3);
    x0 << 0.5, -0.3, 0.0;

    auto result = planner.plan(x0);
    ASSERT_TRUE(result.converged);

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3);
    A(0, 1) = dt;
    A(0, 2) = 0.5 * dt * dt;
    A(1, 2) = dt;

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 1);
    B(0, 0) = (1.0 / 6.0) * dt * dt * dt;
    B(1, 0) = 0.5 * dt * dt;
    B(2, 0) = dt;

    for (int k = 0; k < cfg.N; ++k) {
        Eigen::VectorXd x_next = A * result.x[k] + B * result.u[k];
        EXPECT_NEAR(x_next(0), result.x[k + 1](0), 1e-8)
            << "Position dynamics violated at step " << k;
        EXPECT_NEAR(x_next(1), result.x[k + 1](1), 1e-8)
            << "Velocity dynamics violated at step " << k;
        EXPECT_NEAR(x_next(2), result.x[k + 1](2), 1e-8)
            << "Acceleration dynamics violated at step " << k;
    }
}

// ============================================================================
// Test 5: High lateral velocity toward boundary → controller corrects
// ============================================================================
TEST(LateralPlannerTest, ActiveBoundaryAvoidance) {
    auto planner = makePlanner(/*lane_half_width=*/1.75,
                               /*max_lat_vel=*/1.5,
                               /*max_lat_acc=*/3.0,
                               /*max_lat_jerk=*/10.0,
                               /*N=*/40);
    const auto& cfg = planner.plannerConfig();
    const auto& sc  = planner.solverConfig();
    const double hw  = cfg.lane_half_width;
    const double tol = sc.eps_pri;

    // Start near center but with high lateral velocity toward right boundary
    Eigen::VectorXd x0(3);
    x0 << 0.5, 1.2, 0.0;  // moving fast toward the right

    auto result = planner.plan(x0);

    EXPECT_TRUE(result.converged);

    // States must not exceed lane boundaries
    for (int k = 0; k <= cfg.N; ++k) {
        EXPECT_GE(result.x[k](0), -hw - tol)
            << "Boundary violated at step " << k << ": y=" << result.x[k](0);
        EXPECT_LE(result.x[k](0), hw + tol)
            << "Boundary violated at step " << k << ": y=" << result.x[k](0);
    }

    // The controller must decelerate at some point (negative jerk to reduce ay)
    bool has_negative_jerk = false;
    for (int k = 0; k < cfg.N; ++k) {
        if (result.u[k](0) < -0.5) {
            has_negative_jerk = true;
            break;
        }
    }
    EXPECT_TRUE(has_negative_jerk)
        << "Expected negative jerk to avoid boundary";
}

// ============================================================================
// Test 6: Timing fields are populated
// ============================================================================
TEST(LateralPlannerTest, TimingPopulated) {
    auto planner = makePlanner();

    Eigen::VectorXd x0(3);
    x0 << 0.5, -0.3, 0.0;

    auto result = planner.plan(x0);

    EXPECT_GT(result.time_kkt_us, 0.0) << "KKT timing should be positive";
    EXPECT_GT(result.time_solve_us, 0.0) << "Solve timing should be positive";
}
