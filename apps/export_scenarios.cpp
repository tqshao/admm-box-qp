// Runs lateral planner scenarios loaded from JSON config and exports
// trajectory data as CSV.  Usage:  export_scenarios [output_dir] [config_path]

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "admm/lateral_planner.h"
#include <nlohmann/json.hpp>

using namespace admm;

// ---------------------------------------------------------------------------
// Timing accumulator
// ---------------------------------------------------------------------------
struct TimingRecord {
    std::string name;
    double kkt_us;
    double solve_us;
    int iterations;
    bool converged;
    double final_rho;
};
static std::vector<TimingRecord> g_timing;

// ---------------------------------------------------------------------------
// JSON → config helpers  (override defaults with per-scenario fields)
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
    for (const auto& o : j) {
        obs.push_back({o["y_lo"], o["y_hi"], o["k_start"], o["k_end"]});
    }
    return obs;
}

static Eigen::VectorXd makeX0(const nlohmann::json& j) {
    Eigen::VectorXd x0(j.size());
    for (int i = 0; i < (int)j.size(); ++i) x0(i) = j[i];
    return x0;
}

// ---------------------------------------------------------------------------
// CSV writers (4-panel: y, vy, ay, jerk)
// ---------------------------------------------------------------------------
static void write_csv(const std::string& path,
                      const PlannerConfig& cfg,
                      const ADMMResult& result) {
    const int N = cfg.N;
    const double dt = cfg.dt;

    std::ofstream f(path);
    f << "time,y,vy,ay,jerk,y_min,y_max,vy_min,vy_max,ay_min,ay_max,jerk_min,jerk_max\n";

    for (int k = 0; k <= N; ++k) {
        double t = k * dt;
        double jerk = (k < N) ? result.u[k](0) : 0.0;
        f << t << ","
          << result.x[k](0) << ","
          << result.x[k](1) << ","
          << result.x[k](2) << ","
          << jerk << ","
          << -cfg.lane_half_width << "," << cfg.lane_half_width << ","
          << -cfg.max_lat_vel     << "," << cfg.max_lat_vel     << ","
          << -cfg.max_lat_acc     << "," << cfg.max_lat_acc     << ","
          << -cfg.max_lat_jerk    << "," << cfg.max_lat_jerk    << "\n";
    }
    f << "# converged=" << result.converged
      << " iterations=" << result.iterations
      << " kkt_us=" << result.time_kkt_us
      << " solve_us=" << result.time_solve_us << "\n";
}

static void write_csv_obstacle(const std::string& path,
                               const PlannerConfig& cfg,
                               const ADMMResult& result,
                               const std::vector<ObstacleRegion>& obstacles) {
    const int N = cfg.N;
    const double dt = cfg.dt;

    LateralPlanner tmp(cfg, {});
    auto [lo, hi] = tmp.buildObstacleBounds(obstacles);

    std::ofstream f(path);
    f << "time,y,vy,ay,jerk,y_min,y_max,vy_min,vy_max,ay_min,ay_max,jerk_min,jerk_max\n";

    const int stride = 4;  // nx + nu = 3 + 1
    for (int k = 0; k <= N; ++k) {
        double t = k * dt;
        double jerk = (k < N) ? result.u[k](0) : 0.0;
        int x_idx = (k < N) ? k * stride : N * stride;

        f << t << ","
          << result.x[k](0) << ","
          << result.x[k](1) << ","
          << result.x[k](2) << ","
          << jerk << ","
          << lo(x_idx)     << "," << hi(x_idx)     << ","
          << lo(x_idx + 1) << "," << hi(x_idx + 1) << ","
          << lo(x_idx + 2) << "," << hi(x_idx + 2) << ","
          << (k < N ? lo(x_idx + 3) : -cfg.max_lat_jerk) << ","
          << (k < N ? hi(x_idx + 3) :  cfg.max_lat_jerk) << "\n";
    }
    for (const auto& obs : obstacles) {
        f << "# obstacle y_lo=" << obs.y_lo << " y_hi=" << obs.y_hi
          << " k=[" << obs.k_start << "," << obs.k_end << "]\n";
    }
    f << "# converged=" << result.converged
      << " iterations=" << result.iterations
      << " kkt_us=" << result.time_kkt_us
      << " solve_us=" << result.time_solve_us << "\n";
}

// ---------------------------------------------------------------------------
// Merge defaults with per-scenario overrides
// ---------------------------------------------------------------------------
static nlohmann::json merge_override(const nlohmann::json& defaults,
                                     const nlohmann::json& overrides) {
    nlohmann::json merged = defaults;
    if (!overrides.is_null()) {
        merged.update(overrides);
    }
    return merged;
}

// ---------------------------------------------------------------------------
// Run one scenario
// ---------------------------------------------------------------------------
static void run_scenario(const std::string& csv_dir,
                         const nlohmann::json& scenario,
                         const nlohmann::json& default_planner,
                         const nlohmann::json& default_solver) {
    std::string name = scenario["name"];
    auto pc = makePlanner(merge_override(default_planner, scenario.value("planner", nlohmann::json())));
    auto sc = makeSolver(merge_override(default_solver, scenario.value("solver", nlohmann::json())));
    auto x0 = makeX0(scenario["x0"]);
    auto obs = makeObstacles(scenario.value("obstacles", nlohmann::json::array()));

    LateralPlanner planner(pc, sc);

    ADMMResult result;
    if (obs.empty()) {
        result = planner.plan(x0);
    } else {
        result = planner.plan(x0, obs);
    }

    g_timing.push_back({name, result.time_kkt_us, result.time_solve_us,
                         result.iterations, result.converged, result.final_rho});

    std::string csv_path = csv_dir + "/" + name + ".csv";
    if (obs.empty()) {
        write_csv(csv_path, pc, result);
    } else {
        write_csv_obstacle(csv_path, pc, result, obs);
    }

    std::cout << "  -> " << csv_path
              << "  (conv=" << result.converged
              << " it=" << result.iterations
              << " kkt=" << result.time_kkt_us << "us"
              << " solve=" << result.time_solve_us << "us)\n";
}

// ---------------------------------------------------------------------------
// Timing table
// ---------------------------------------------------------------------------
static void print_timing_table() {
    std::cout << "\n" << std::string(78, '=') << "\n";
    std::cout << std::left << std::setw(22) << "Scenario"
              << std::right
              << std::setw(8) << "Conv" << std::setw(8) << "Iters"
              << std::setw(14) << "KKT [us]" << std::setw(14) << "Solve [us]"
              << std::setw(12) << "Total [ms]" << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (const auto& r : g_timing) {
        double total_ms = (r.kkt_us + r.solve_us) / 1000.0;
        std::cout << std::left << std::setw(22) << r.name << std::right
                  << std::setw(8) << (r.converged ? "Y" : "N")
                  << std::setw(8) << r.iterations
                  << std::setw(14) << std::fixed << std::setprecision(1) << r.kkt_us
                  << std::setw(14) << r.solve_us
                  << std::setw(12) << std::setprecision(3) << total_ms << "\n";
    }
    std::cout << std::string(78, '=') << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string out_dir = "figures";
    if (argc > 1) out_dir = argv[1];

    std::string config_path = "../configs/scenarios.json";
    if (argc > 2) config_path = argv[2];

    // Create subdirectories: csv/ and png/ under output dir
    std::string csv_dir = out_dir + "/csv";
    std::string png_dir = out_dir + "/png";
    std::filesystem::create_directories(csv_dir);
    std::filesystem::create_directories(png_dir);

    // Load scenario definitions from JSON
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

    for (const auto& sc : scenarios) {
        run_scenario(csv_dir, sc, default_planner, default_solver);
    }

    print_timing_table();

    // Write timing CSV for the plot script
    {
        std::string timing_path = csv_dir + "/_timing.csv";
        std::ofstream tf(timing_path);
        tf << "name,converged,iterations,kkt_us,solve_us,total_ms,final_rho\n";
        for (const auto& r : g_timing) {
            double total_ms = (r.kkt_us + r.solve_us) / 1000.0;
            tf << r.name << ","
               << (r.converged ? 1 : 0) << ","
               << r.iterations << ","
               << r.kkt_us << ","
               << r.solve_us << ","
               << total_ms << ","
               << r.final_rho << "\n";
        }
        std::cout << "  -> " << timing_path << "\n";
    }

    std::cout << "\nCSV files saved to " << csv_dir << "/\n";
    std::cout << "PNG plots go to " << png_dir << "/\n";
    return 0;
}
