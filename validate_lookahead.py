
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from scenario_testing import (
    ROAD_TYPES,
    TEST_SPEEDS,
    MAX_LATERAL_ACCELERATION,
    MAX_ACCELERATION,
    MAX_BRAKING,
    WHEELBASE,
    DT,
    MAX_STEPS,
    EVALUATION_START,
    EVALUATION_END,
    BIN_SIZE,
    create_path,
    calculate_spatial_metrics,
)

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.steering_rate_limiter import SteeringRateLimiter
from src.controllers.pid_speed import PIDSpeedController
from src.planning.speed_planner import SpeedPlanner


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

INPUT_CSV = (
    ROOT
    / "results"
    / "lookahead_optimization"
    / "lookahead_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "lookahead_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_LOOKAHEAD = 2.0
BASELINE_GAIN = 0.6
MIN_SHARED_COVERAGE = 0.95


# ==========================================
# 2. LOAD OPTIMIZED PARAMETER
# ==========================================

def load_best_gain():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "Missing lookahead_results.csv. "
            "Run optimize_lookahead.py first."
        )

    candidates = []

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            gain = float(row["gain"])
            rmse = float(row["path_rmse"])
            coverage = float(
                row["common_coverage"]
            )

            if np.isfinite(gain) and np.isfinite(rmse):
                candidates.append({
                    "gain": gain,
                    "rmse": rmse,
                    "coverage": coverage
                })

    if not candidates:
        raise RuntimeError(
            "No valid lookahead candidates found."
        )

    # Use the lowest-RMSE candidate.
    # The validation experiment will test
    # whether it generalizes.
    best = min(
        candidates,
        key=lambda item: item["rmse"]
    )

    print(
        f"Optimized candidate: k={best['gain']:.2f}"
    )

    print(
        f"Training RMSE: {best['rmse']:.4f} m"
    )

    if best["coverage"] < MIN_SHARED_COVERAGE:
        print(
            "WARNING: Training common coverage "
            "is below 95%."
        )

    return best["gain"]


# ==========================================
# 3. RUN ONE VEHICLE SIMULATION
# ==========================================

def run_vehicle(
    path_x,
    path_y,
    path_distance,
    speed_profile,
    gain
):

    vehicle = BicycleModel(
        wheelbase=WHEELBASE
    )

    vehicle.x = 0.0
    vehicle.y = -2.0
    vehicle.yaw = 0.0
    vehicle.velocity = 0.0

    lateral = PurePursuitController(
        wheelbase=WHEELBASE
    )

    limiter = SteeringRateLimiter(
        max_rate_deg=60.0
    )

    longitudinal = PIDSpeedController(
        kp=1.0,
        ki=0.1,
        kd=0.05,
        min_acceleration=-MAX_BRAKING,
        max_acceleration=MAX_ACCELERATION,
        anti_windup=True
    )

    history = {
        "time": [],
        "distance": [],
        "speed": [],
        "target_speed": [],
        "path_error": [],
        "lateral_acceleration": [],
        "x": [],
        "y": [],
        "steering": []
    }

    last_index = 0
    stop_reason = "Maximum steps"

    for step in range(MAX_STEPS):

        distances = np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )

        nearest_index = (
            last_index
            + int(
                np.argmin(
                    distances[last_index:]
                )
            )
        )

        last_index = nearest_index

        if nearest_index >= len(path_x) - 1:
            stop_reason = "Reached final path index"
            break

        target_speed = float(
            speed_profile[nearest_index]
        )

        lookahead = (
            BASE_LOOKAHEAD
            + gain * abs(vehicle.velocity)
        )

        target_index = nearest_index

        while target_index < len(path_x) - 1:

            distance_to_target = np.hypot(
                path_x[target_index] - vehicle.x,
                path_y[target_index] - vehicle.y
            )

            if distance_to_target >= lookahead:
                break

            target_index += 1

        desired_steering = lateral.calculate_steering(
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            path_x[target_index],
            path_y[target_index]
        )

        steering = limiter.update(
            desired_steering,
            DT
        )

        acceleration = longitudinal.update(
            target_speed=target_speed,
            current_speed=vehicle.velocity,
            dt=DT
        )

        lateral_acceleration = (
            vehicle.velocity ** 2
            / WHEELBASE
            * np.tan(steering)
        )

        history["time"].append(step * DT)
        history["distance"].append(
            path_distance[nearest_index]
        )
        history["speed"].append(
            vehicle.velocity
        )
        history["target_speed"].append(
            target_speed
        )
        history["path_error"].append(
            distances[nearest_index]
        )
        history["lateral_acceleration"].append(
            lateral_acceleration
        )
        history["x"].append(vehicle.x)
        history["y"].append(vehicle.y)
        history["steering"].append(
            np.degrees(steering)
        )

        vehicle.update(
            acceleration=acceleration,
            steering_angle=steering,
            dt=DT
        )

        if not np.all(np.isfinite([
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            vehicle.velocity
        ])):
            raise RuntimeError(
                f"Invalid state at step {step}"
            )

    for key in history:
        history[key] = np.asarray(
            history[key],
            dtype=float
        )

    history["stop_reason"] = stop_reason

    return history


# ==========================================
# 4. COMPARE ON COMMON SPATIAL BINS
# ==========================================

def compare_histories(
    baseline_history,
    optimized_history
):

    baseline = calculate_spatial_metrics(
        baseline_history
    )

    optimized = calculate_spatial_metrics(
        optimized_history
    )

    shared = (
        baseline["valid"]
        & optimized["valid"]
    )

    shared_coverage = float(
        np.mean(shared)
    )

    if not np.any(shared):
        return None, shared_coverage

    def metrics(data):

        return {
            "path_rmse": float(
                np.sqrt(
                    np.mean(
                        data["error_rms"][shared] ** 2
                    )
                )
            ),
            "max_lateral_acceleration": float(
                np.max(
                    data["lateral_max"][shared]
                )
            ),
            "speed_rmse": float(
                np.sqrt(
                    np.mean(
                        data["speed_error_rms"][shared] ** 2
                    )
                )
            )
        }

    return {
        "baseline": metrics(baseline),
        "optimized": metrics(optimized),
        "baseline_coverage": baseline["coverage"],
        "optimized_coverage": optimized["coverage"]
    }, shared_coverage


# ==========================================
# 5. RUN CROSS-SCENARIO VALIDATION
# ==========================================

def run_validation(best_gain):

    results = []

    scenario_number = 0

    for road_name, config in ROAD_TYPES.items():

        path_x, path_y = create_path(
            config["amplitude"],
            config["frequency"]
        )

        for max_speed in TEST_SPEEDS:

            scenario_number += 1

            print(
                f"\nScenario {scenario_number}/12: "
                f"{road_name}, {max_speed:.1f} m/s"
            )

            planner = SpeedPlanner(
                max_speed=max_speed,
                max_lateral_acceleration=(
                    MAX_LATERAL_ACCELERATION
                ),
                max_acceleration=MAX_ACCELERATION,
                max_braking=MAX_BRAKING
            )

            plan = planner.plan(
                path_x,
                path_y
            )

            speed_profile = np.asarray(
                plan["speed_profile"],
                dtype=float
            )

            path_distance = np.asarray(
                plan["distance"],
                dtype=float
            )

            baseline_history = run_vehicle(
                path_x,
                path_y,
                path_distance,
                speed_profile,
                BASELINE_GAIN
            )

            optimized_history = run_vehicle(
                path_x,
                path_y,
                path_distance,
                speed_profile,
                best_gain
            )

            comparison, shared_coverage = (
                compare_histories(
                    baseline_history,
                    optimized_history
                )
            )

            if comparison is None:
                print(
                    "WARNING: No shared spatial bins."
                )
                continue

            baseline_rmse = comparison[
                "baseline"
            ]["path_rmse"]

            optimized_rmse = comparison[
                "optimized"
            ]["path_rmse"]

            improvement = (
                (baseline_rmse - optimized_rmse)
                / baseline_rmse * 100.0
                if baseline_rmse > 1e-12
                else 0.0
            )

            reliable = (
                shared_coverage >= MIN_SHARED_COVERAGE
            )

            results.append({
                "scenario": scenario_number,
                "road": road_name,
                "max_speed": max_speed,
                "baseline_gain": BASELINE_GAIN,
                "optimized_gain": best_gain,
                "baseline_rmse": baseline_rmse,
                "optimized_rmse": optimized_rmse,
                "improvement_percent": improvement,
                "baseline_lateral": comparison[
                    "baseline"
                ]["max_lateral_acceleration"],
                "optimized_lateral": comparison[
                    "optimized"
                ]["max_lateral_acceleration"],
                "baseline_speed_rmse": comparison[
                    "baseline"
                ]["speed_rmse"],
                "optimized_speed_rmse": comparison[
                    "optimized"
                ]["speed_rmse"],
                "baseline_coverage": comparison[
                    "baseline_coverage"
                ],
                "optimized_coverage": comparison[
                    "optimized_coverage"
                ],
                "shared_coverage": shared_coverage,
                "reliable_coverage": reliable
            })

            print(
                f"Baseline RMSE: {baseline_rmse:.4f} m"
            )
            print(
                f"Optimized RMSE: {optimized_rmse:.4f} m"
            )
            print(
                f"Improvement: {improvement:+.2f}%"
            )
            print(
                f"Shared coverage: "
                f"{shared_coverage * 100:.1f}%"
            )

    return results


# ==========================================
# 6. SAVE RESULTS
# ==========================================

def save_results(results):

    if not results:
        raise RuntimeError(
            "No valid validation results."
        )

    csv_path = (
        OUTPUT_DIR / "validation_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    json_path = (
        OUTPUT_DIR / "validation_results.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nCSV saved:", csv_path)
    print("JSON saved:", json_path)


# ==========================================
# 7. PLOT RESULTS
# ==========================================

def plot_results(results):

    roads = list(ROAD_TYPES.keys())

    # Figure 1: RMSE by road and speed
    plt.figure(figsize=(12, 6))

    for road in roads:

        subset = [
            r for r in results
            if r["road"] == road
        ]

        x = [r["max_speed"] for r in subset]

        plt.plot(
            x,
            [r["baseline_rmse"] for r in subset],
            marker="o",
            linestyle="--",
            label=f"{road} Baseline"
        )

        plt.plot(
            x,
            [r["optimized_rmse"] for r in subset],
            marker="o",
            label=f"{road} Optimized"
        )

    plt.xlabel("Maximum Speed (m/s)")
    plt.ylabel("Path RMSE (m)")
    plt.title("Cross-Scenario Path Tracking")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "validation_rmse.png",
        dpi=200
    )

    # Figure 2: Improvement percentage
    plt.figure(figsize=(12, 6))

    for road in roads:

        subset = [
            r for r in results
            if r["road"] == road
        ]

        plt.plot(
            [r["max_speed"] for r in subset],
            [r["improvement_percent"] for r in subset],
            marker="o",
            label=road
        )

    plt.axhline(
        0.0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Maximum Speed (m/s)")
    plt.ylabel("RMSE Improvement (%)")
    plt.title("Optimized vs Baseline Improvement")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "validation_improvement.png",
        dpi=200
    )

    # Figure 3: Lateral acceleration
    plt.figure(figsize=(12, 6))

    for road in roads:

        subset = [
            r for r in results
            if r["road"] == road
        ]

        x = [r["max_speed"] for r in subset]

        plt.plot(
            x,
            [r["baseline_lateral"] for r in subset],
            marker="o",
            linestyle="--",
            label=f"{road} Baseline"
        )

        plt.plot(
            x,
            [r["optimized_lateral"] for r in subset],
            marker="o",
            label=f"{road} Optimized"
        )

    plt.axhline(
        MAX_LATERAL_ACCELERATION,
        color="red",
        linestyle=":",
        label="Planning Limit"
    )

    plt.xlabel("Maximum Speed (m/s)")
    plt.ylabel("Max Lateral Acceleration (m/s^2)")
    plt.title("Lateral Acceleration Validation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "validation_lateral.png",
        dpi=200
    )

    # Figure 4: Shared coverage
    plt.figure(figsize=(12, 6))

    for road in roads:

        subset = [
            r for r in results
            if r["road"] == road
        ]

        plt.plot(
            [r["max_speed"] for r in subset],
            [r["shared_coverage"] * 100 for r in subset],
            marker="o",
            label=road
        )

    plt.axhline(
        MIN_SHARED_COVERAGE * 100,
        color="red",
        linestyle="--",
        label="95% Threshold"
    )

    plt.xlabel("Maximum Speed (m/s)")
    plt.ylabel("Shared Coverage (%)")
    plt.title("Validation Coverage")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "validation_coverage.png",
        dpi=200
    )

    plt.show()


# ==========================================
# 8. SUMMARY
# ==========================================

def print_summary(results, best_gain):

    reliable = [
        r for r in results
        if r["reliable_coverage"]
    ]

    print("\n" + "=" * 60)
    print("CROSS-SCENARIO VALIDATION SUMMARY")
    print("=" * 60)

    print(f"Baseline gain: {BASELINE_GAIN:.2f}")
    print(f"Optimized gain: {best_gain:.2f}")
    print(f"Scenarios completed: {len(results)}/12")
    print(
        f"Scenarios with >=95% shared coverage: "
        f"{len(reliable)}/12"
    )

    if not reliable:
        print(
            "No scenarios meet the coverage threshold."
        )
        print(
            "Do not draw a generalization conclusion."
        )
        return

    wins = sum(
        r["optimized_rmse"] < r["baseline_rmse"]
        for r in reliable
    )

    mean_improvement = np.mean([
        r["improvement_percent"]
        for r in reliable
    ])

    print(
        f"Optimized RMSE wins: "
        f"{wins}/{len(reliable)}"
    )

    print(
        f"Mean RMSE improvement: "
        f"{mean_improvement:+.2f}%"
    )

    if len(reliable) < len(results):
        print(
            "WARNING: Some scenarios were excluded "
            "from the summary due to low coverage."
        )


# ==========================================
# 9. MAIN
# ==========================================

def main():

    print("=" * 60)
    print("MILESTONE 9 - CROSS-SCENARIO VALIDATION")
    print("=" * 60)

    best_gain = load_best_gain()

    results = run_validation(best_gain)

    save_results(results)
    print_summary(results, best_gain)
    plot_results(results)

    print("\nValidation completed.")
    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
