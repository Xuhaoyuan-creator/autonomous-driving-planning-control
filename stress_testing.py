
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.steering_rate_limiter import SteeringRateLimiter
from src.controllers.pid_speed import PIDSpeedController
from src.planning.speed_planner import SpeedPlanner


# ==========================================
# CONFIGURATION
# ==========================================

DT = 0.1
MAX_STEPS = 1500
WHEELBASE = 2.7

MAX_ACCELERATION = 2.0
MAX_BRAKING = 3.0
MAX_LATERAL_ACCELERATION = 0.5

BASE_LOOKAHEAD = 2.0
LOOKAHEAD_GAIN = 0.6

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "stress_testing"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# TEST SCENARIOS
# ==========================================

SCENARIOS = [
    {
        "name": "Normal",
        "amplitude": 2.0,
        "frequency": 0.05,
        "max_speed": 8.0,
        "initial_y": 0.0,
        "initial_yaw_deg": 0.0,
    },
    {
        "name": "Large Offset",
        "amplitude": 2.0,
        "frequency": 0.05,
        "max_speed": 8.0,
        "initial_y": -5.0,
        "initial_yaw_deg": 0.0,
    },
    {
        "name": "Heading Error",
        "amplitude": 2.0,
        "frequency": 0.05,
        "max_speed": 8.0,
        "initial_y": 0.0,
        "initial_yaw_deg": 20.0,
    },
    {
        "name": "Sharp Curve",
        "amplitude": 5.0,
        "frequency": 0.10,
        "max_speed": 8.0,
        "initial_y": 0.0,
        "initial_yaw_deg": 0.0,
    },
    {
        "name": "Low Speed",
        "amplitude": 2.0,
        "frequency": 0.05,
        "max_speed": 2.0,
        "initial_y": 0.0,
        "initial_yaw_deg": 0.0,
    },
    {
        "name": "Combined Stress",
        "amplitude": 5.0,
        "frequency": 0.10,
        "max_speed": 8.0,
        "initial_y": -5.0,
        "initial_yaw_deg": 20.0,
    },
]


# ==========================================
# CREATE ROAD
# ==========================================

def create_path(amplitude, frequency):

    path_x = np.linspace(
        0.0, 250.0, 1251
    )

    path_y = (
        amplitude
        * np.sin(frequency * path_x)
    )

    return path_x, path_y


# ==========================================
# SIMULATION
# ==========================================

def run_scenario(config):

    path_x, path_y = create_path(
        config["amplitude"],
        config["frequency"]
    )

    planner = SpeedPlanner(
        max_speed=config["max_speed"],
        max_lateral_acceleration=(
            MAX_LATERAL_ACCELERATION
        ),
        max_braking=MAX_BRAKING,
        max_acceleration=MAX_ACCELERATION
    )

    plan = planner.plan(
        path_x,
        path_y
    )

    speed_profile = np.asarray(
        plan["speed_profile"],
        dtype=float
    )

    vehicle = BicycleModel(
        wheelbase=WHEELBASE
    )

    vehicle.x = 0.0
    vehicle.y = config["initial_y"]
    vehicle.yaw = np.radians(
        config["initial_yaw_deg"]
    )
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
        "x": [],
        "y": [],
        "time": [],
        "speed": [],
        "path_error": [],
        "lateral_acceleration": [],
        "steering_deg": [],
    }

    last_index = 0
    stop_reason = "Maximum steps"
    numerical_failure = False

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
            + LOOKAHEAD_GAIN
            * abs(vehicle.velocity)
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

        history["x"].append(vehicle.x)
        history["y"].append(vehicle.y)
        history["time"].append(step * DT)
        history["speed"].append(
            vehicle.velocity
        )
        history["path_error"].append(
            distances[nearest_index]
        )
        history["lateral_acceleration"].append(
            abs(lateral_acceleration)
        )
        history["steering_deg"].append(
            np.degrees(steering)
        )

        vehicle.update(
            acceleration=acceleration,
            steering_angle=steering,
            dt=DT
        )

        state = np.array([
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            vehicle.velocity,
            steering,
            acceleration
        ])

        if not np.all(np.isfinite(state)):
            numerical_failure = True
            stop_reason = "Numerical failure"
            break

    for key in history:
        history[key] = np.asarray(
            history[key],
            dtype=float
        )

    if len(history["path_error"]) == 0:
        raise RuntimeError(
            "Simulation produced no data."
        )

    # Evaluate tracking after initial recovery.
    # Exclude the first 10 seconds.
    evaluation_mask = (
        history["time"] >= 10.0
    )

    if not np.any(evaluation_mask):
        evaluation_mask = np.ones(
            len(history["time"]),
            dtype=bool
        )

    evaluated_errors = (
        history["path_error"][evaluation_mask]
    )

    path_rmse = float(
        np.sqrt(
            np.mean(evaluated_errors ** 2)
        )
    )

    max_path_error = float(
        np.max(evaluated_errors)
    )

    max_lateral = float(
        np.max(
            history["lateral_acceleration"]
        )
    )

    final_index = last_index

    completion = (
        final_index
        / (len(path_x) - 1)
        * 100.0
    )

    # These are preliminary engineering thresholds,
    # not universal safety requirements.
    completed = completion >= 95.0

    stable = (
        not numerical_failure
        and path_rmse < 3.0
        and max_path_error < 8.0
    )

    passed = stable and completed

    metrics = {
        "scenario": config["name"],
        "path_rmse": path_rmse,
        "max_path_error": max_path_error,
        "max_lateral_acceleration": max_lateral,
        "completion_percent": completion,
        "stable": stable,
        "completed": completed,
        "passed": passed,
        "stop_reason": stop_reason
    }

    return (
        metrics,
        history,
        path_x,
        path_y
    )


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(results):

    csv_path = (
        RESULTS_DIR / "stress_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", csv_path)


# ==========================================
# PLOT RESULTS
# ==========================================

def plot_results(all_runs):

    # Figure 1: Vehicle trajectories
    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 9)
    )

    for ax, run in zip(
        axes.flat,
        all_runs
    ):

        metrics, history, path_x, path_y = run

        ax.plot(
            path_x,
            path_y,
            "k--",
            label="Reference"
        )

        ax.plot(
            history["x"],
            history["y"],
            label="Vehicle"
        )

        ax.set_title(
            metrics["scenario"]
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "stress_trajectories.png",
        dpi=200
    )

    # Figure 2: Path RMSE
    names = [
        run[0]["scenario"]
        for run in all_runs
    ]

    rmse = [
        run[0]["path_rmse"]
        for run in all_runs
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(names, rmse)

    plt.ylabel("Path RMSE (m)")
    plt.title("Stress Test Path Tracking")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "stress_rmse.png",
        dpi=200
    )

    # Figure 3: Completion
    completion = [
        run[0]["completion_percent"]
        for run in all_runs
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        names,
        completion
    )

    plt.axhline(
        95.0,
        color="red",
        linestyle="--",
        label="Completion threshold"
    )

    plt.ylabel("Completion (%)")
    plt.title("Stress Test Path Completion")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.ylim(0, 105)
    plt.grid(
        axis="y",
        alpha=0.3
    )
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "stress_completion.png",
        dpi=200
    )

    plt.show()


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("MILESTONE 10 - ROBUSTNESS STRESS TEST")
    print("=" * 60)

    results = []
    all_runs = []

    for index, config in enumerate(
        SCENARIOS,
        start=1
    ):

        print(
            f"\nRunning {index}/6: "
            f"{config['name']}"
        )

        run = run_scenario(config)

        metrics = run[0]

        results.append(metrics)
        all_runs.append(run)

        print(
            f"Path RMSE: "
            f"{metrics['path_rmse']:.3f} m"
        )

        print(
            f"Max path error: "
            f"{metrics['max_path_error']:.3f} m"
        )

        print(
            f"Max lateral acceleration: "
            f"{metrics['max_lateral_acceleration']:.3f} m/s^2"
        )

        print(
            f"Completion: "
            f"{metrics['completion_percent']:.1f}%"
        )

        print(
            f"Status: "
            f"{'PASS' if metrics['passed'] else 'FAIL'}"
        )

        print(
            f"Stop reason: "
            f"{metrics['stop_reason']}"
        )

    save_results(results)
    plot_results(all_runs)

    passed = sum(
        result["passed"]
        for result in results
    )

    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)

    print(f"Scenarios passed: {passed}/6")

    for result in results:

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"{result['scenario']}: {status}"
        )

    print("\nResults directory:", RESULTS_DIR)


if __name__ == "__main__":
    main()
