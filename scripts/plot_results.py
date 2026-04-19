#!/usr/bin/env python3
"""Read CSV files exported by export_scenarios and generate PNG plots."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


TITLES = {
    "01_lane_keeping_center":       "Scenario 1: Lane Keeping from Center",
    "02_lane_keeping_offset":       "Scenario 2: Lane Keeping from Offset",
    "03_near_boundary":             "Scenario 3: Near Boundary",
    "04_offset_with_velocity":      "Scenario 4: Offset with Velocity",
    "05_active_boundary_avoidance": "Scenario 5: Active Boundary Avoidance",
    "06_swerve_left":               "Scenario 6: Obstacle on Right \u2192 Swerve Left",
    "07_swerve_right":              "Scenario 7: Obstacle on Left \u2192 Swerve Right",
    "08_s_curve":                   "Scenario 8: S-Curve (Right then Left Obstacle)",
    "09_narrow_gap":                "Scenario 9: Narrow Gap (Thread Through)",
}


def read_csv(path):
    """Read a scenario CSV, return data dict and obstacle list."""
    data = {"time": [], "y": [], "vy": [], "ay": [], "jerk": [],
            "y_min": [], "y_max": [], "vy_min": [], "vy_max": [],
            "ay_min": [], "ay_max": [], "jerk_min": [], "jerk_max": []}
    obstacles = []

    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if line.startswith("# obstacle"):
                    parts = line.split()
                    obs = {}
                    for p in parts[2:]:
                        k, v = p.split("=", 1)
                        if k in ("y_lo", "y_hi"):
                            obs[k] = float(v)
                        elif k == "k":
                            vals = v.lstrip("[").rstrip("]").split(",")
                            obs["k_start"] = int(vals[0])
                            obs["k_end"] = int(vals[1])
                    if obs:
                        obstacles.append(obs)
                continue
            if not line:
                continue
            vals = line.split(",")
            row = {h: float(v) for h, v in zip(header, vals)}
            data["time"].append(row["time"])
            data["y"].append(row["y"])
            data["vy"].append(row["vy"])
            data["ay"].append(row["ay"])
            data["jerk"].append(row["jerk"])
            data["y_min"].append(row["y_min"])
            data["y_max"].append(row["y_max"])
            data["vy_min"].append(row["vy_min"])
            data["vy_max"].append(row["vy_max"])
            data["ay_min"].append(row["ay_min"])
            data["ay_max"].append(row["ay_max"])
            data["jerk_min"].append(row["jerk_min"])
            data["jerk_max"].append(row["jerk_max"])

    for k in data:
        data[k] = np.array(data[k])
    return data, obstacles


def read_timing(path):
    """Read the _timing.csv summary file."""
    import csv
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_scenario(name, data, obstacles, out_dir, dt=0.1):
    """Generate a 4-panel plot for one scenario."""
    t = data["time"]

    # If obstacles exist, make a 5-panel plot (top-view + 4 time-series)
    has_obs = len(obstacles) > 0
    n_panels = 5 if has_obs else 4
    heights = [1.2] + [1] * 4 if has_obs else [1] * 4

    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.8 * n_panels),
                             sharex=False, gridspec_kw={"height_ratios": heights})
    fig.suptitle(TITLES.get(name, name), fontsize=14, fontweight="bold")

    panel_idx = 0

    # --- Top-view panel (obstacle scenarios only) ---
    if has_obs:
        ax = axes[panel_idx]
        panel_idx += 1

        # Plot trajectory in x-y plane (time → longitudinal distance)
        x_long = data["time"] * 20.0  # v_x = 20 m/s
        ax.plot(x_long, data["y"], "b-", linewidth=1.5, label="ego trajectory")
        ax.plot(x_long[0], data["y"][0], "go", markersize=8, label="start")

        # Lane boundaries
        hw = data["y_max"][0]
        ax.axhline(hw, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(-hw, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)

        # Draw obstacle rectangles
        for obs in obstacles:
            x0 = obs["k_start"] * dt * 20.0
            x1 = (obs["k_end"] + 1) * dt * 20.0
            rect = mpatches.Rectangle(
                (x0, obs["y_lo"]), x1 - x0, obs["y_hi"] - obs["y_lo"],
                linewidth=1, edgecolor="red", facecolor="red", alpha=0.3,
                label="obstacle" if obs == obstacles[0] else None)
            ax.add_patch(rect)

        ax.set_ylabel("y [m]")
        ax.set_xlabel("Longitudinal distance [m]")
        ax.set_title("Top View", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    # --- Panel: Lateral position ---
    ax = axes[panel_idx]
    panel_idx += 1
    ax.plot(t, data["y"], "b-o", markersize=3, linewidth=1.2, label="y (position)")
    ax.fill_between(t, data["y_min"], data["y_max"],
                    color="green", alpha=0.08, label="feasible region")
    ax.plot(t, data["y_min"], "g--", linewidth=0.6, alpha=0.6)
    ax.plot(t, data["y_max"], "g--", linewidth=0.6, alpha=0.6)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    if has_obs:
        for obs in obstacles:
            t0 = obs["k_start"] * dt
            t1 = (obs["k_end"] + 1) * dt
            ax.axvspan(t0, t1, color="red", alpha=0.10)
            ax.fill_between([t0, t1], obs["y_lo"], obs["y_hi"],
                            color="red", alpha=0.25,
                            label="obstacle" if obs == obstacles[0] else None)
    ax.set_ylabel("Position [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel: Lateral velocity ---
    ax = axes[panel_idx]
    panel_idx += 1
    ax.plot(t, data["vy"], "g-o", markersize=3, linewidth=1.2, label="vy (velocity)")
    ax.fill_between(t, data["vy_min"], data["vy_max"], color="green", alpha=0.05)
    ax.axhline(data["vy_max"][0], color="r", linestyle="--", linewidth=1.0,
               label=f"limit \u00b1{abs(data['vy_max'][0]):.1f} m/s")
    ax.axhline(data["vy_min"][0], color="r", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    if has_obs:
        for obs in obstacles:
            t0, t1 = obs["k_start"] * dt, (obs["k_end"] + 1) * dt
            ax.axvspan(t0, t1, color="red", alpha=0.10)
    ax.set_ylabel("Velocity [m/s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel: Lateral acceleration ---
    ax = axes[panel_idx]
    panel_idx += 1
    ax.plot(t, data["ay"], "m-o", markersize=3, linewidth=1.2, label="ay (accel)")
    ax.fill_between(t, data["ay_min"], data["ay_max"], color="green", alpha=0.05)
    ax.axhline(data["ay_max"][0], color="r", linestyle="--", linewidth=1.0,
               label=f"limit \u00b1{abs(data['ay_max'][0]):.1f} m/s\u00b2")
    ax.axhline(data["ay_min"][0], color="r", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    if has_obs:
        for obs in obstacles:
            t0, t1 = obs["k_start"] * dt, (obs["k_end"] + 1) * dt
            ax.axvspan(t0, t1, color="red", alpha=0.10)
    ax.set_ylabel("Accel [m/s\u00b2]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel: Lateral jerk ---
    ax = axes[panel_idx]
    t_u = t[:-1]
    jerk = data["jerk"][:-1]
    jerk_min = data["jerk_min"][:-1]
    jerk_max = data["jerk_max"][:-1]
    ax.step(t_u, jerk, where="post", color="orange", linewidth=1.2,
            label="jerk (control)")
    ax.fill_between(t_u, jerk_min, jerk_max, color="green", alpha=0.05)
    ax.axhline(jerk_max[0], color="r", linestyle="--", linewidth=1.0,
               label=f"limit \u00b1{abs(jerk_max[0]):.1f} m/s\u00b3")
    ax.axhline(jerk_min[0], color="r", linestyle="--", linewidth=1.0)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    if has_obs:
        for obs in obstacles:
            t0, t1 = obs["k_start"] * dt, (obs["k_end"] + 1) * dt
            ax.axvspan(t0, t1, color="red", alpha=0.10)
    ax.set_ylabel("Jerk [m/s\u00b3]")
    ax.set_xlabel("Time [s]")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_timing(timing_path, out_dir):
    """Generate a timing comparison bar chart."""
    rows = read_timing(timing_path)
    if not rows:
        return

    names = [r["name"][:18] for r in rows]
    kkt = [float(r["kkt_us"]) for r in rows]
    solve = [float(r["solve_us"]) for r in rows]
    iters = [int(r["iterations"]) for r in rows]
    converged = [r["converged"] == "1" for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Solver Timing & Iteration Summary", fontsize=14, fontweight="bold")

    # --- Left: stacked bar chart of KKT + solve time ---
    x = np.arange(len(names))
    width = 0.6

    bars1 = ax1.bar(x, kkt, width, label="KKT factorization", color="#4ECDC4")
    bars2 = ax1.bar(x, solve, width, bottom=kkt, label="ADMM solve", color="#FF6B6B")

    # Mark non-converged with red border
    for i, conv in enumerate(converged):
        if not conv:
            ax1.patches[i].set_edgecolor("red")
            ax1.patches[i].set_linewidth(2)
            ax1.patches[i + len(names)].set_edgecolor("red")
            ax1.patches[i + len(names)].set_linewidth(2)

    ax1.set_ylabel("Time [us]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=9)
    ax1.set_title("KKT + Solve Time")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add total time labels
    for i in range(len(names)):
        total = kkt[i] + solve[i]
        ax1.text(i, total + 200, f"{total/1000:.1f}ms",
                ha="center", va="bottom", fontsize=7)

    # --- Right: iteration count ---
    colors = ["#4ECDC4" if c else "#FF6B6B" for c in converged]
    ax2.bar(x, iters, width, color=colors)
    ax2.set_ylabel("Iterations")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_title("ADMM Iterations (green=converged, red=not)")
    ax2.grid(True, alpha=0.3, axis="y")

    for i, it in enumerate(iters):
        ax2.text(i, it + 30, str(it), ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "_timing_summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot lateral planner scenario results.")
    parser.add_argument("--data-dir", default="figures/csv",
                        help="Directory containing CSV files")
    parser.add_argument("--out-dir", default="figures/png",
                        help="Directory to save PNG plots")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step for obstacle shading [s]")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csvs = sorted(f for f in os.listdir(args.data_dir)
                  if f.endswith(".csv") and not f.startswith("_"))
    if not csvs:
        print(f"No CSV files found in {args.data_dir}/")
        sys.exit(1)

    for csv_name in csvs:
        name = csv_name.replace(".csv", "")
        path = os.path.join(args.data_dir, csv_name)
        print(f"Plotting {name} ...")
        data, obstacles = read_csv(path)
        plot_scenario(name, data, obstacles, args.out_dir, args.dt)

    # Timing summary
    timing_path = os.path.join(args.data_dir, "_timing.csv")
    if os.path.exists(timing_path):
        print("\nPlotting timing summary ...")
        plot_timing(timing_path, args.out_dir)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
