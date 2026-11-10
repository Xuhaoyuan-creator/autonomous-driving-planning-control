
from pathlib import Path
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.steering_rate_limiter import SteeringRateLimiter
from src.controllers.pid_speed import PIDSpeedController
from src.planning.speed_planner import SpeedPlanner


# ==================================================
# 1. CONFIGURATION
# ==================================================

DT = 0.1
MAX_STEPS = 1200

WHEELBASE = 2.7

MAX_ACCELERATION = 2.0
MAX_BRAKING = 3.0
MAX_LATERAL_ACCELERATION = 0.5

PATH_LENGTH = 250.0
PATH_POINTS = 1251

# Compare the same road interval.
EVALUATION_START = 20.0
EVALUATION_END = 200.0
BIN_SIZE = 2.0

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "scenario_testing"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 2. TEST SCENARIOS
# ==================================================

ROAD_TYPES = {
    "Gentle": {
        "amplitude": 2.0,
        "frequency": 0.05
    },
    "Medium": {
        "amplitude": 3.5,
        "frequency": 0.065
    },
    "Sharp": {
        "amplitude": 5.0,
        "frequency": 0.08
    }
}

TEST_SPEEDS = [6.0, 8.0, 10.0, 12.0]


# ==================================================
# 3. CREATE REFERENCE PATH
# ==================================================

def create_path(amplitude, frequency):

    path_x = np.linspace(
        0.0,
        PATH_LENGTH,
        PATH_POINTS
    )

    path_y = (
        amplitude
        * np.sin(frequency * path_x)
    )

    return path_x, path_y


# ==================================================
# 4. SIMULATION
# ==================================================

def run_simulation(
    path_x,
    path_y,
    planned_speed,
    path_distance,
    max_speed,
    use_speed_planning
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
        "y": []
    }

    last_index = 0
    stop_reason = "Maximum steps"

    for step in range(MAX_STEPS):

        # ----------------------------------
        # A. Find nearest path point
        # ----------------------------------

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

        # ----------------------------------
        # B. Check endpoint
        # ----------------------------------

        if nearest_index == len(path_x) - 1:
            stop_reason = "Reached final path index"
            break

        # ----------------------------------
        # C. Select target speed
        # ----------------------------------

        if use_speed_planning:
            target_speed = float(
                planned_speed[nearest_index]
            )
        else:
            target_speed = max_speed

        # ----------------------------------
        # D. Adaptive lookahead
        # ----------------------------------

        lookahead = (
            lateral.calculate_lookahead(
                vehicle.velocity
            )
        )

        target_index = nearest_index

        while target_index < len(path_x) - 1:

            target_distance = np.hypot(
                path_x[target_index] - vehicle.x,
                path_y[target_index] - vehicle.y
            )

            if target_distance >= lookahead:
                break

            target_index += 1

        # ----------------------------------
        # E. Lateral control
        # ----------------------------------

        desired_steering = (
            lateral.calculate_steering(
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                path_x[target_index],
                path_y[target_index]
            )
        )

        steering = limiter.update(
            desired_steering,
            DT
        )

        # ----------------------------------
        # F. Longitudinal control
        # ----------------------------------

        acceleration = longitudinal.update(
            target_speed=target_speed,
            current_speed=vehicle.velocity,
            dt=DT
        )

        # ----------------------------------
        # G. Lateral acceleration
        # ----------------------------------

        lateral_acceleration = (
            vehicle.velocity ** 2
            / WHEELBASE
            * np.tan(steering)
        )

        # ----------------------------------
        # H. Record data
        # ----------------------------------

        history["time"].append(
            step * DT
        )

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

        # ----------------------------------
        # I. Update vehicle
        # ----------------------------------

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
                "Invalid vehicle state"
            )

    for key in history:
        history[key] = np.asarray(
            history[key],
            dtype=float
        )

    history["stop_reason"] = stop_reason

    return history


# ==================================================
# 5. SPATIAL EVALUATION
# ==================================================

def calculate_spatial_metrics(history):

    s = history["distance"]

    path_error = history["path_error"]

    lateral = np.abs(
        history["lateral_acceleration"]
    )

    speed_error = (
        history["target_speed"]
        - history["speed"]
    )

    edges = np.arange(
        EVALUATION_START,
        EVALUATION_END + BIN_SIZE,
        BIN_SIZE
    )

    bin_count = len(edges) - 1

    error_rms = np.full(
        bin_count, np.nan
    )

    lateral_max = np.full(
        bin_count, np.nan
    )

    speed_error_rms = np.full(
        bin_count, np.nan
    )

    for i in range(bin_count):

        mask = (
            (s >= edges[i])
            & (s < edges[i + 1])
        )

        if not np.any(mask):
            continue

        error_rms[i] = np.sqrt(
            np.mean(
                path_error[mask] ** 2
            )
        )

        lateral_max[i] = np.max(
            lateral[mask]
        )

        speed_error_rms[i] = np.sqrt(
            np.mean(
                speed_error[mask] ** 2
            )
        )

    valid = np.isfinite(error_rms)

    coverage = float(np.mean(valid))

    return {
        "error_rms": error_rms,
        "lateral_max": lateral_max,
        "speed_error_rms": speed_error_rms,
        "valid": valid,
        "coverage": coverage
    }


# ==================================================
# 6. COMPARE ON SHARED ROAD BINS
# ==================================================

def summarize_shared_metrics(
    fixed,
    planned
):

    shared = (
        fixed["valid"]
        & planned["valid"]
    )

    shared_coverage = float(
        np.mean(shared)
    )

    results = {}

    for name, data in [
        ("Fixed", fixed),
        ("Planned", planned)
    ]:

        if not np.any(shared):

            results[name] = {
                "path_rmse": np.nan,
                "max_lateral_acceleration": np.nan,
                "speed_rmse": np.nan
            }

            continue

        results[name] = {
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

    return results, shared_coverage


# ==================================================
# 7. RUN ALL SCENARIOS
# ==================================================

def run_all_scenarios():

    all_results = []

    scenario_number = 0

    for road_name, road_config in ROAD_TYPES.items():

        path_x, path_y = create_path(
            road_config["amplitude"],
            road_config["frequency"]
        )

        for max_speed in TEST_SPEEDS:

            scenario_number += 1

            print(
                f"\nScenario {scenario_number}/12"
            )

            print(
                f"Road: {road_name}"
            )

            print(
                f"Maximum Speed: {max_speed} m/s"
            )

            # ------------------------------
            # Generate speed plan
            # ------------------------------

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

            planned_speed = np.asarray(
                plan["speed_profile"]
            )

            path_distance = np.asarray(
                plan["distance"]
            )

            # ------------------------------
            # Fixed speed simulation
            # ------------------------------

            fixed_history = run_simulation(
                path_x=path_x,
                path_y=path_y,
                planned_speed=planned_speed,
                path_distance=path_distance,
                max_speed=max_speed,
                use_speed_planning=False
            )

            # ------------------------------
            # Planned speed simulation
            # ------------------------------

            planned_history = run_simulation(
                path_x=path_x,
                path_y=path_y,
                planned_speed=planned_speed,
                path_distance=path_distance,
                max_speed=max_speed,
                use_speed_planning=True
            )

            fixed_spatial = (
                calculate_spatial_metrics(
                    fixed_history
                )
            )

            planned_spatial = (
                calculate_spatial_metrics(
                    planned_history
                )
            )

            comparison, shared_coverage = (
                summarize_shared_metrics(
                    fixed_spatial,
                    planned_spatial
                )
            )

            for mode, spatial, history in [
                ("Fixed", fixed_spatial, fixed_history),
                ("Planned", planned_spatial, planned_history)
            ]:

                metrics = comparison[mode]

                all_results.append({
                    "scenario": scenario_number,
                    "road": road_name,
                    "max_speed": max_speed,
                    "mode": mode,
                    "coverage": spatial["coverage"],
                    "shared_coverage": shared_coverage,
                    "path_rmse": metrics["path_rmse"],
                    "max_lateral_acceleration":
                        metrics["max_lateral_acceleration"],
                    "speed_rmse": metrics["speed_rmse"],
                    "minimum_planned_speed":
                        float(np.min(planned_speed)),
                    "stop_reason":
                        history["stop_reason"]
                })

            print(
                f"Shared coverage: "
                f"{shared_coverage * 100:.1f}%"
            )

            print(
                f"Fixed RMSE: "
                f"{comparison['Fixed']['path_rmse']:.3f} m"
            )

            print(
                f"Planned RMSE: "
                f"{comparison['Planned']['path_rmse']:.3f} m"
            )

    return all_results


# ==================================================
# 8. SAVE CSV
# ==================================================

def save_csv(results):

    output = (
        RESULTS_DIR / "scenario_results.csv"
    )

    with output.open(
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

    print("\nCSV saved:", output)


# ==================================================
# 9. SAVE JSON
# ==================================================

def save_json(results):

    output = (
        RESULTS_DIR / "scenario_results.json"
    )

    with output.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            allow_nan=False,
            default=lambda x: (
                None
                if isinstance(x, float) and np.isnan(x)
                else x
            )
        )

    print("JSON saved:", output)


# ==================================================
# 10. PLOT RESULTS
# ==================================================

def plot_results(results):

    roads = list(ROAD_TYPES.keys())

    # --------------------------------------
    # Figure 1: Path RMSE
    # --------------------------------------

    plt.figure(figsize=(12, 6))

    for road in roads:

        for mode in ["Fixed", "Planned"]:

            subset = [
                row for row in results
                if row["road"] == road
                and row["mode"] == mode
            ]

            x = [
                row["max_speed"]
                for row in subset
            ]

            y = [
                row["path_rmse"]
                for row in subset
            ]

            plt.plot(
                x,
                y,
                marker="o",
                label=f"{road} - {mode}"
            )

    plt.xlabel("Maximum Speed (m/s)")
    plt.ylabel("Path RMSE (m)")
    plt.title("Path Tracking Across Scenarios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "scenario_path_rmse.png",
        dpi=200
    )

    # --------------------------------------
    # Figure 2: Lateral acceleration
    # --------------------------------------

    plt.figure(figsize=(12, 6))

    for road in roads:

        for mode in ["Fixed", "Planned"]:

            subset = [
                row for row in results
                if row["road"] == road
                and row["mode"] == mode
            ]

            x = [
                row["max_speed"]
                for row in subset
            ]

            y = [
                row["max_lateral_acceleration"]
                for row in subset
            ]

            plt.plot(
                x,
                y,
                marker="o",
                label=f"{road} - {mode}"
            )

    plt.axhline(
        MAX_LATERAL_ACCELERATION,
        color="red",
        linestyle="--",
        label="Planning Limit"
    )

    plt.xlabel("Maximum Speed (m/s)")
    plt.ylabel("Maximum Lateral Acceleration (m/s^2)")
    plt.title("Lateral Acceleration Across Scenarios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "scenario_lateral_accel.png",
        dpi=200
    )

    # --------------------------------------
    # Figure 3: Coverage
    # --------------------------------------

    plt.figure(figsize=(12, 6))

    for road in roads:

        subset = [
            row for row in results
            if row["road"] == road
            and row["mode"] == "Planned"
        ]

        x = [
            row["max_speed"]
            for row in subset
        ]

        y = [
            row["shared_coverage"] * 100
            for row in subset
        ]

        plt.plot(
            x,
            y,
            marker="o",
            label=road
        )

    plt.xlabel("Maximum Speed (m/s)")
    plt.ylabel("Shared Coverage (%)")
    plt.title("Shared Road Coverage")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "scenario_coverage.png",
        dpi=200
    )

    plt.show()


# ==================================================
# 11. MAIN
# ==================================================

def main():

    print("=" * 60)
    print("AUTOMATED SCENARIO TESTING")
    print("=" * 60)

    results = run_all_scenarios()

    save_csv(results)

    # Convert NaN to None for strict JSON.
    json_results = []

    for row in results:

        converted = {}

        for key, value in row.items():

            if (
                isinstance(value, (float, np.floating))
                and not np.isfinite(value)
            ):
                converted[key] = None
            else:
                converted[key] = value

        json_results.append(converted)

    save_json(json_results)

    plot_results(results)

    print("\nAll scenarios completed.")
    print("Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
