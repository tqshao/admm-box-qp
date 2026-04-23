// Benchmark: runs ADMM (with/without polish) and OSQP on all scenarios
// and prints a side-by-side comparison table with pre/post-polish metrics.
//
// Usage: benchmark_solvers [config_path]

#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "admm/lateral_planner.h"
#include "admm/osqp_solver.h"
#include <nlohmann/json.hpp>

using namespace admm;

// ---------------------------------------------------------------------------
// JSON -> config helpers (same as export_scenarios.cpp)
// ---------------------------------------------------------------------------
static SolverConfig makeSolver(const nlohmann::json& j) {
    SolverConfig s;
    if (j.contains("rho"))            s.rho            = j["rho"];
    if (j.contains("alpha"))          s.alpha          = j["alpha"];
    if (j.contains("eps_abs"))        s.eps_abs        = j["eps_abs"];
    if (j.contains("eps_rel"))       s.eps_rel       = j["eps_rel"];
    if (j.contains("max_iter"))       s.max_iter       = j["max_iter"];
    if (j.contains("kkt_reg"))        s.kkt_reg        = j["kkt_reg"];
    if (j.contains("adaptive_rho"))    s.adaptive_rho    = j["adaptive_rho"];
    if (j.contains("adapt_interval"))  s.adapt_interval  = j["adapt_interval"];
    if (j.contains("adapt_tolerance")) s.adapt_tolerance = j["adapt_tolerance"];
    if (j.contains("rho_min"))         s.rho_min         = j["rho_min"];
    if (j.contains("rho_max"))         s.rho_max         = j["rho_max"];
    if (j.contains("use_riccati"))     s.use_riccati     = j["use_riccati"];
    if (j.contains("polish"))          s.polish          = j["polish"];
    if (j.contains("polish_delta"))    s.polish_delta    = j["polish_delta"];
    if (j.contains("polish_refine_iter")) s.polish_refine_iter = j["polish_refine_iter"];
    return s;
}

static PlannerConfig makePlanner(const nlohmann::json& j) {
    PlannerConfig p;
    if (j.contains("v_x"))             p.v_x             = j["v_x"];
    if (j.contains("dt"))              p.dt              = j["dt"];
    if (j.contains("N"))               p.N               = j["N"];
    if (j.contains("lane_half_width")) p.lane_half_width = j["lane_half_width"];
    if (j.contains("max_lat_vel"))     p.max_lat_vel     = j["max_lat_vel"];
    if (j.contains("max_lat_acc"))     p.max_lat_acc     = j["max_lat_acc"];
    if (j.contains("max_lat_jerk"))    p.max_lat_jerk    = j["max_lat_jerk"];
    if (j.contains("q_y"))             p.q_y             = j["q_y"];
    if (j.contains("q_vy"))            p.q_vy            = j["q_vy"];
    if (j.contains("q_ay"))            p.q_ay            = j["q_ay"];
    if (j.contains("r"))               p.r               = j["r"];
    if (j.contains("p_y"))             p.p_y             = j["p_y"];
    if (j.contains("p_vy"))            p.p_vy            = j["p_vy"];
    if (j.contains("p_ay"))            p.p_ay            = j["p_ay"];
    return p;
}

static std::vector<ObstacleRegion> makeObstacles(const nlohmann::json& j) {
    std::vector<ObstacleRegion> obs;
    for (const auto& o : j)
        obs.push_back({o["y_lo"], o["y_hi"], o["k_start"], o["k_end"]});
    return obs;
}

static Eigen::VectorXd makeX0(const nlohmann::json& j) {
    Eigen::VectorXd x0(j.size());
    for (int i = 0; i < (int)j.size(); ++i) x0(i) = j[i];
    return x0;
}

static nlohmann::json merge_override(const nlohmann::json& defaults,
                                     const nlohmann::json& overrides) {
    nlohmann::json merged = defaults;
    if (!overrides.is_null()) merged.update(overrides);
    return merged;
}

// ---------------------------------------------------------------------------
// Compute objective cost from trajectory
// ---------------------------------------------------------------------------
static double computeCost(const ProblemData& data,
                          const std::vector<Eigen::VectorXd>& x,
                          const std::vector<Eigen::VectorXd>& u) {
    double cost = 0.0;
    for (int k = 0; k < data.N; ++k) {
        cost += 0.5 * x[k].dot(data.Q * x[k]);
        cost += 0.5 * u[k].dot(data.R * u[k]);
    }
    cost += 0.5 * x[data.N].dot(data.P * x[data.N]);
    return cost;
}

// ---------------------------------------------------------------------------
// Benchmark record
// ---------------------------------------------------------------------------
struct BenchRecord {
    std::string name;
    // ADMM (with polishing enabled)
    int    admm_iters;
    bool   admm_converged;
    bool   admm_polished;
    double admm_solve_us, admm_polish_us;
    // Pre-polish ADMM metrics
    double pre_cost, pre_violation, pre_pri_res, pre_dua_res;
    // Post-polish (final) ADMM metrics
    double post_cost, post_violation;
    // OSQP
    double osqp_setup_us, osqp_solve_us;
    int    osqp_iters;
    bool   osqp_converged;
    double osqp_pri_res, osqp_dua_res;
    double cost_osqp;
};

// ---------------------------------------------------------------------------
// Run one scenario
// ---------------------------------------------------------------------------
static void run_benchmark(const nlohmann::json& scenario,
                          const nlohmann::json& default_planner,
                          const nlohmann::json& default_solver,
                          std::vector<BenchRecord>& records) {
    std::string name = scenario["name"];
    auto pc = makePlanner(merge_override(default_planner, scenario.value("planner", nlohmann::json())));
    auto sc = makeSolver(merge_override(default_solver, scenario.value("solver", nlohmann::json())));
    auto x0 = makeX0(scenario["x0"]);
    auto obs = makeObstacles(scenario.value("obstacles", nlohmann::json::array()));

    std::cout << "  Scenario: " << name << " ..." << std::flush;

    BenchRecord rec;
    rec.name = name;

    // Build common ProblemData
    LateralPlanner planner(pc, sc);
    ProblemData pd = planner.problemData();
    pd.x0 = x0;
    if (!obs.empty()) {
        auto [lo, hi] = planner.buildObstacleBounds(obs);
        pd.custom_lower_bounds = lo;
        pd.custom_upper_bounds = hi;
    }

    // --- ADMM (with polishing) ---
    pd.use_riccati = false;
    ADMMSolver admm(pd);
    auto admm_res = admm.solve();
    rec.admm_iters     = admm_res.iterations;
    rec.admm_converged = admm_res.converged;
    rec.admm_polished  = admm_res.polished;
    rec.admm_solve_us  = admm_res.time_solve_us;
    rec.admm_polish_us = admm_res.time_polish_us;
    rec.pre_cost       = admm_res.pre_polish_objective_cost;
    rec.pre_violation  = admm_res.pre_polish_max_bound_violation;
    rec.pre_pri_res    = admm_res.pre_polish_primal_residual;
    rec.pre_dua_res    = admm_res.pre_polish_dual_residual;
    rec.post_cost      = admm_res.objective_cost;
    rec.post_violation = admm_res.max_bound_violation;

    // --- OSQP ---
    pd.use_riccati = false;
    OsqpSolver osqp(pd);
    auto osqp_res = osqp.solve();
    rec.osqp_setup_us  = osqp_res.time_setup_us;
    rec.osqp_solve_us  = osqp_res.time_solve_us;
    rec.osqp_iters     = osqp_res.iterations;
    rec.osqp_converged = osqp_res.converged;
    rec.osqp_pri_res   = osqp_res.primal_residual;
    rec.osqp_dua_res   = osqp_res.dual_residual;
    rec.cost_osqp      = computeCost(pd, osqp_res.x, osqp_res.u);

    records.push_back(rec);

    std::cout << "  ADMM(" << (rec.admm_converged ? "Y" : "N")
              << "/" << rec.admm_iters << " it"
              << (rec.admm_polished ? " polished" : "") << ")"
              << "  OSQP(" << (rec.osqp_converged ? "Y" : "N")
              << "/" << rec.osqp_iters << " it)\n";
}

// ---------------------------------------------------------------------------
// Print per-scenario comparison + summary
// ---------------------------------------------------------------------------
static void print_results(const std::vector<BenchRecord>& records) {
    // --- Per-scenario detail ---
    const int W = 80;
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "                  Per-Scenario Detail\n";
    std::cout << std::string(W, '=') << "\n";

    for (const auto& r : records) {
        double admm_total_ms = r.admm_solve_us / 1000.0;
        double osqp_total_ms = (r.osqp_setup_us + r.osqp_solve_us) / 1000.0;

        std::cout << "\n--- " << r.name << " ---\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Metric            |  ADMM(pre)   |  ADMM(post)  |    OSQP\n";
        std::cout << "  ------------------+--------------+--------------+------------\n";
        std::cout << "  Converged         |" << std::setw(12) << (r.admm_converged ? "Yes" : "No") << "  |"
                  << std::setw(12) << (r.admm_converged ? (r.admm_polished ? "Polished" : "Yes") : "N/A") << "  |"
                  << std::setw(10) << (r.osqp_converged ? "Yes" : "No") << "\n";
        std::cout << "  Iterations        |" << std::setw(12) << r.admm_iters << "  |"
                  << std::setw(12) << "" << "  |"
                  << std::setw(10) << r.osqp_iters << "\n";
        std::cout << "  Primal res        |" << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.pre_pri_res << "  |"
                  << std::setw(12) << "" << "  |"
                  << std::setw(10) << r.osqp_pri_res << "\n";
        std::cout << "  Dual res          |" << std::setw(12) << r.pre_dua_res << "  |"
                  << std::setw(12) << "" << "  |"
                  << std::setw(10) << r.osqp_dua_res << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Objective cost    |" << std::setw(12) << r.pre_cost << "  |"
                  << std::setw(12) << r.post_cost << "  |"
                  << std::setw(10) << std::setprecision(6) << r.cost_osqp << "\n";
        std::cout << "  Bound violation   |" << std::scientific << std::setprecision(2)
                  << std::setw(12) << r.pre_violation << "  |"
                  << std::setw(12) << r.post_violation << "  |"
                  << std::setw(10) << "" << "\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Solve time (us)   |" << std::setprecision(1) << std::setw(12) << r.admm_solve_us << "  |"
                  << std::setw(12) << "" << "  |"
                  << std::setw(10) << r.osqp_solve_us << "\n";
        if (r.admm_polished) {
            std::cout << "  Polish time (us)  |" << std::setw(12) << "" << "  |"
                      << std::setw(12) << std::setprecision(1) << r.admm_polish_us << "  |"
                      << std::setw(10) << "" << "\n";
        }
    }
    std::cout << "\n" << std::string(W, '=') << "\n";

    // --- Summary table ---
    const int SW = 90;
    std::cout << "\n" << std::string(SW, '=') << "\n";
    std::cout << "              ADMM vs OSQP Summary\n";
    std::cout << std::string(SW, '=') << "\n";

    std::cout << std::left << std::setw(24) << "Scenario"
              << " | "
              << std::right
              << std::setw(5) << "Iter" << " "
              << std::setw(4) << "Pol" << " "
              << std::setw(10) << "Cost" << " "
              << std::setw(10) << "Viol" << " "
              << " | "
              << std::setw(5) << "Iter" << " "
              << std::setw(10) << "Cost"
              << "\n";
    std::cout << std::string(SW, '-') << "\n";

    for (const auto& r : records) {
        std::cout << std::left << std::setw(24) << r.name << " | "
                  << std::right
                  << std::setw(5) << r.admm_iters << " "
                  << std::setw(4) << (r.admm_polished ? "*" : " ")
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << r.post_cost << " "
                  << std::scientific << std::setprecision(2)
                  << std::setw(10) << r.post_violation << " "
                  << " | "
                  << std::setw(5) << r.osqp_iters << " "
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << r.cost_osqp
                  << "\n";
    }

    std::cout << std::string(SW, '=') << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string config_path = "../configs/scenarios.json";
    if (argc > 1) config_path = argv[1];

    std::ifstream cf(config_path);
    if (!cf.is_open()) {
        std::cerr << "Error: cannot open scenario config " << config_path << "\n";
        return 1;
    }
    nlohmann::json root = nlohmann::json::parse(cf);

    auto default_planner = root.value("defaults", nlohmann::json::object())
                                 .value("planner", nlohmann::json::object());
    auto default_solver  = root.value("defaults", nlohmann::json::object())
                                 .value("solver", nlohmann::json::object());
    auto scenarios = root.value("scenarios", nlohmann::json::array());

    std::cout << "Loaded " << scenarios.size() << " scenarios from "
              << config_path << "\n\n";

    std::vector<BenchRecord> records;
    for (const auto& sc : scenarios)
        run_benchmark(sc, default_planner, default_solver, records);

    print_results(records);

    return 0;
}
