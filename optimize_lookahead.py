
from pathlib import Path
import csv

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

MAX_SPEED = 10.0
MAX_ACCELERATION = 2.0
MAX_BRAKING = 3.0
MAX_LATERAL_ACCELERATION = 0.5

BASE_LOOKAHEAD = 2.0
LOOKAHEAD_GAINS = [0.2, 0.4, 0.6, 0.8, 1.0]

EVALUATION_START = 20.0
EVALUATION_END = 200.0
BIN_SIZE = 2.0

PATH_X = np.linspace(0.0, 250.0, 1251)
PATH_Y = 5.0 * np.sin(0.08 * PATH_X)

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "lookahead_optimization"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. SPEED PLANNING
# ==================================================

planner = SpeedPlanner(
    max_speed=MAX_SPEED,
    max_lateral_acceleration=MAX_LATERAL_ACCELERATION,
    max_acceleration=MAX_ACCELERATION,
    max_braking=MAX_BRAKING
)

plan = planner.plan(PATH_X, PATH_Y)

PLANNED_SPEED = np.asarray(
    plan["speed_profile"], dtype=float
)

PATH_DISTANCE = np.asarray(
    plan["distance"], dtype=float
)


# ==================================================
# 3. SIMULATION
# ==================================================

def run_simulation(lookahead_gain):

    vehicle = BicycleModel(wheelbase=WHEELBASE)

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
        "x": [],
        "y": [],
        "speed": [],
        "target_speed": [],
        "path_error": [],
        "lateral_acceleration": [],
        "steering": [],
        "lookahead": []
    }

    last_index = 0
    stop_reason = "Maximum steps"

    for step in range(MAX_STEPS):

        # A. Find nearest point
        distances = np.hypot(
            PATH_X - vehicle.x,
            PATH_Y - vehicle.y
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

        if nearest_index >= len(PATH_X) - 1:
            stop_reason = "Reached final path index"
            break

        # B. Target speed
        target_speed = float(
            PLANNED_SPEED[nearest_index]
        )

        # C. Custom adaptive lookahead
        lookahead = (
            BASE_LOOKAHEAD
            + lookahead_gain * abs(vehicle.velocity)
        )

        # D. Find lookahead point
        target_index = nearest_index

        while target_index < len(PATH_X) - 1:

            target_distance = np.hypot(
                PATH_X[target_index] - vehicle.x,
                PATH_Y[target_index] - vehicle.y
            )

            if target_distance >= lookahead:
                break

            target_index += 1

        # E. Steering control
        desired_steering = lateral.calculate_steering(
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            PATH_X[target_index],
            PATH_Y[target_index]
        )

        steering = limiter.update(
            desired_steering,
            DT
        )

        # F. Longitudinal control
        acceleration = longitudinal.update(
            target_speed=target_speed,
            current_speed=vehicle.velocity,
            dt=DT
        )

        # G. Lateral acceleration
        lateral_acceleration = (
            vehicle.velocity ** 2
            / WHEELBASE
            * np.tan(steering)
        )

        # H. Record data
        history["time"].append(step * DT)
        history["distance"].append(
            PATH_DISTANCE[nearest_index]
        )
        history["x"].append(vehicle.x)
        history["y"].append(vehicle.y)
        history["speed"].append(vehicle.velocity)
        history["target_speed"].append(target_speed)
        history["path_error"].append(
            distances[nearest_index]
        )
        history["lateral_acceleration"].append(
            lateral_acceleration
        )
        history["steering"].append(
            np.degrees(steering)
        )
        history["lookahead"].append(lookahead)

        # I. Update vehicle
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
                f"Invalid vehicle state at step {step}"
            )

    for key in history:
        history[key] = np.asarray(
            history[key], dtype=float
        )

    history["stop_reason"] = stop_reason

    return history


# ==================================================
# 4. SPATIAL METRICS
# ==================================================

def evaluate_history(history):

    edges = np.arange(
        EVALUATION_START,
        EVALUATION_END + BIN_SIZE,
        BIN_SIZE
    )

    n_bins = len(edges) - 1

    binned_error = np.full(n_bins, np.nan)
    binned_lateral = np.full(n_bins, np.nan)

    s = history["distance"]

    for i in range(n_bins):

        mask = (
            (s >= edges[i])
            & (s < edges[i + 1])
        )

        if not np.any(mask):
            continue

        binned_error[i] = np.sqrt(
            np.mean(
                history["path_error"][mask] ** 2
            )
        )

        binned_lateral[i] = np.max(
            np.abs(
                history["lateral_acceleration"][mask]
            )
        )

    valid = np.isfinite(binned_error)

    return {
        "error": binned_error,
        "lateral": binned_lateral,
        "valid": valid,
        "coverage": float(np.mean(valid)),
        "centers": (edges[:-1] + edges[1:]) / 2
    }


# ==================================================
# 5. MAIN
# ==================================================

def main():

    print("=" * 60)
    print("ADAPTIVE LOOKAHEAD OPTIMIZATION")
    print("=" * 60)

    histories = {}
    evaluations = {}

    # Run all candidates
    for gain in LOOKAHEAD_GAINS:

        print(f"\nTesting lookahead gain: {gain:.1f}")

        history = run_simulation(gain)
        evaluation = evaluate_history(history)

        histories[gain] = history
        evaluations[gain] = evaluation

        print(
            f"Coverage: "
            f"{evaluation['coverage'] * 100:.1f}%"
        )
        print(
            f"Stop reason: {history['stop_reason']}"
        )

    # Only compare spatial bins visited by all gains.
    common_mask = np.ones(
        len(next(iter(evaluations.values()))["valid"]),
        dtype=bool
    )

    for evaluation in evaluations.values():
        common_mask &= evaluation["valid"]

    common_coverage = float(
        np.mean(common_mask)
    )

    if not np.any(common_mask):
        raise RuntimeError(
            "No common road bins among candidates."
        )

    print(
        f"\nCommon coverage: "
        f"{common_coverage * 100:.1f}%"
    )

    results = []

    for gain in LOOKAHEAD_GAINS:

        evaluation = evaluations[gain]

        path_rmse = float(
            np.sqrt(
                np.mean(
                    evaluation["error"][common_mask] ** 2
                )
            )
        )

        max_lateral = float(
            np.max(
                evaluation["lateral"][common_mask]
            )
        )

        results.append({
            "gain": gain,
            "coverage": evaluation["coverage"],
            "common_coverage": common_coverage,
            "path_rmse": path_rmse,
            "max_lateral_acceleration": max_lateral
        })

        print(
            f"Gain={gain:.1f} | "
            f"RMSE={path_rmse:.3f} m | "
            f"Max Lat Accel={max_lateral:.3f} m/s^2"
        )

    # Lowest path error among candidates
    best = min(
        results,
        key=lambda row: row["path_rmse"]
    )

    print("\n=== BEST CANDIDATE ===")
    print(
        f"Gain: {best['gain']:.1f}"
    )
    print(
        f"Path RMSE: {best['path_rmse']:.3f} m"
    )

    if common_coverage < 0.95:
        print(
            "WARNING: Common coverage below 95%."
        )
        print(
            "Best candidate applies only to shared bins."
        )

    if best["max_lateral_acceleration"] > MAX_LATERAL_ACCELERATION:
        print(
            "WARNING: Best candidate exceeds "
            "the lateral acceleration planning limit."
        )

    # Save CSV
    csv_path = (
        RESULTS_DIR / "lookahead_results.csv"
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

    # Figure 1: Path RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(
        [r["gain"] for r in results],
        [r["path_rmse"] for r in results],
        marker="o"
    )
    plt.xlabel("Lookahead Speed Gain")
    plt.ylabel("Path RMSE (m)")
    plt.title("Lookahead Gain vs Path Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "lookahead_rmse.png",
        dpi=200
    )

    # Figure 2: Lateral acceleration
    plt.figure(figsize=(10, 5))
    plt.plot(
        [r["gain"] for r in results],
        [r["max_lateral_acceleration"] for r in results],
        marker="o"
    )
    plt.axhline(
        MAX_LATERAL_ACCELERATION,
        color="red",
        linestyle="--",
        label="Planning Limit"
    )
    plt.xlabel("Lookahead Speed Gain")
    plt.ylabel("Maximum Lateral Acceleration (m/s^2)")
    plt.title("Lookahead Gain vs Lateral Acceleration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "lookahead_lateral.png",
        dpi=200
    )

    # Figure 3: Actual trajectories
    plt.figure(figsize=(12, 6))
    plt.plot(
        PATH_X, PATH_Y,
        "k--",
        linewidth=2,
        label="Reference Path"
    )

    for gain in LOOKAHEAD_GAINS:
        history = histories[gain]
        plt.plot(
            history["x"],
            history["y"],
            label=f"Gain={gain:.1f}"
        )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory Comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "lookahead_trajectories.png",
        dpi=200
    )

    # Figure 4: Spatial path error
    plt.figure(figsize=(12, 6))

    for gain in LOOKAHEAD_GAINS:
        evaluation = evaluations[gain]
        plt.plot(
            evaluation["centers"],
            np.where(
                common_mask,
                evaluation["error"],
                np.nan
            ),
            label=f"Gain={gain:.1f}"
        )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Path Error RMS (m)")
    plt.title("Spatial Path Error Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "lookahead_spatial_error.png",
        dpi=200
    )

    print("\nResults saved to:", RESULTS_DIR)

    plt.show()


if __name__ == "__main__":
    main()
