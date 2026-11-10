
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.steering_rate_limiter import SteeringRateLimiter
from src.controllers.pid_speed import PIDSpeedController
from src.planning.speed_planner import SpeedPlanner


# ==========================================
# 1. CONFIGURATION
# ==========================================

DT = 0.1
STEPS = 800
WHEELBASE = 2.7

MAX_SPEED = 10.0
MAX_ACCELERATION = 2.0
MAX_BRAKING = 3.0
MAX_LATERAL_ACCELERATION = 0.5

PATH_X = np.linspace(0.0, 250.0, 1251)
PATH_Y = 5.0 * np.sin(0.08 * PATH_X)

RESULTS_DIR = (
    Path(__file__).resolve().parent / "results"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. SPEED PLANNING
# ==========================================

planner = SpeedPlanner(
    max_speed=MAX_SPEED,
    max_lateral_acceleration=MAX_LATERAL_ACCELERATION,
    max_braking=MAX_BRAKING,
    max_acceleration=MAX_ACCELERATION
)

plan = planner.plan(PATH_X, PATH_Y)

PLANNED_SPEED = np.asarray(
    plan["speed_profile"], dtype=float
)

PATH_DISTANCE = np.asarray(
    plan["distance"], dtype=float
)

if len(PLANNED_SPEED) != len(PATH_X):
    raise RuntimeError("Invalid speed profile length")

if not np.all(np.isfinite(PLANNED_SPEED)):
    raise RuntimeError("Invalid planned speed values")


# ==========================================
# 3. SIMULATION
# ==========================================

def run_simulation(use_speed_planning):

    mode = (
        "SPEED PLANNING"
        if use_speed_planning
        else "FIXED SPEED"
    )

    print(f"\n=== RUNNING {mode} ===")

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
        "time": [0.0],
        "x": [vehicle.x],
        "y": [vehicle.y],
        "speed": [vehicle.velocity],
        "target_speed": [],
        "acceleration": [],
        "steering": [],
        "path_error": [],
        "lateral_acceleration": [],
        "distance": []
    }

    last_nearest_index = 0
    stop_reason = "Maximum simulation steps reached"

    for step in range(STEPS):

        # A. Find nearest path point
        distances = np.hypot(
            PATH_X - vehicle.x,
            PATH_Y - vehicle.y
        )

        nearest_index = (
            last_nearest_index
            + int(
                np.argmin(
                    distances[last_nearest_index:]
                )
            )
        )

        last_nearest_index = nearest_index

        # B. Check endpoint
        endpoint_distance = np.hypot(
            vehicle.x - PATH_X[-1],
            vehicle.y - PATH_Y[-1]
        )

        if endpoint_distance <= 2.0:
            stop_reason = "Reached endpoint"
            break

        # Also stop if vehicle passes the path end
        # and is reasonably close to the final point.
        if (
            nearest_index == len(PATH_X) - 1
            and vehicle.x >= PATH_X[-1]
        ):
            stop_reason = "Passed endpoint"
            break

        # C. Target speed
        if use_speed_planning:
            target_speed = float(
                PLANNED_SPEED[nearest_index]
            )
        else:
            target_speed = MAX_SPEED

        # D. Adaptive lookahead
        lookahead = lateral.calculate_lookahead(
            vehicle.velocity
        )

        target_index = nearest_index

        while target_index < len(PATH_X) - 1:

            distance_to_target = np.hypot(
                PATH_X[target_index] - vehicle.x,
                PATH_Y[target_index] - vehicle.y
            )

            if distance_to_target >= lookahead:
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

        # F. Speed control
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

        # H. Store data before update
        history["target_speed"].append(
            target_speed
        )
        history["acceleration"].append(
            acceleration
        )
        history["steering"].append(
            np.degrees(steering)
        )
        history["path_error"].append(
            distances[nearest_index]
        )
        history["lateral_acceleration"].append(
            lateral_acceleration
        )
        history["distance"].append(
            PATH_DISTANCE[nearest_index]
        )

        # I. Vehicle update
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

        # J. Store updated state
        history["time"].append(
            (step + 1) * DT
        )
        history["x"].append(vehicle.x)
        history["y"].append(vehicle.y)
        history["speed"].append(vehicle.velocity)

        if step < 5 or step % 100 == 0:
            print(
                f"{mode} | "
                f"Time={step * DT:.1f}s | "
                f"Index={nearest_index} | "
                f"Target={target_speed:.2f} | "
                f"Actual={history['speed'][-2]:.2f}"
            )

    for key in history:
        history[key] = np.asarray(
            history[key], dtype=float
        )

    history["stop_reason"] = stop_reason

    print(
        f"{mode} finished: "
        f"{len(history['distance'])} steps"
    )
    print("Stop reason:", stop_reason)

    return history


# ==========================================
# 4. PERFORMANCE METRICS
# ==========================================

def calculate_metrics(history):

    if len(history["path_error"]) == 0:
        raise RuntimeError("Empty simulation history")

    speed_error = (
        history["target_speed"]
        - history["speed"][:-1]
    )

    distance_traveled = np.sum(
        np.hypot(
            np.diff(history["x"]),
            np.diff(history["y"])
        )
    )

    return {
        "path_rmse": float(
            np.sqrt(
                np.mean(
                    history["path_error"] ** 2
                )
            )
        ),
        "max_path_error": float(
            np.max(history["path_error"])
        ),
        "speed_rmse": float(
            np.sqrt(
                np.mean(speed_error ** 2)
            )
        ),
        "max_lateral_acceleration": float(
            np.max(
                np.abs(
                    history["lateral_acceleration"]
                )
            )
        ),
        "total_distance": float(
            distance_traveled
        )
    }


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("\n=== SPEED PROFILE ===")
    print(
        f"Minimum: {PLANNED_SPEED.min():.3f} m/s"
    )
    print(
        f"Maximum: {PLANNED_SPEED.max():.3f} m/s"
    )

    fixed = run_simulation(False)
    planned = run_simulation(True)

    fixed_metrics = calculate_metrics(fixed)
    planned_metrics = calculate_metrics(planned)

    print("\n=== EXPERIMENT COMPARISON ===")
    print(
        f"{'Metric':<30}"
        f"{'Fixed':>14}"
        f"{'Planned':>14}"
    )

    for key in fixed_metrics:
        print(
            f"{key:<30}"
            f"{fixed_metrics[key]:>14.3f}"
            f"{planned_metrics[key]:>14.3f}"
        )

    # Figure 1: Path tracking
    plt.figure(figsize=(12, 6))
    plt.plot(
        PATH_X, PATH_Y, "k--",
        label="Reference Path"
    )
    plt.plot(
        fixed["x"], fixed["y"],
        label="Fixed Speed"
    )
    plt.plot(
        planned["x"], planned["y"],
        label="Speed Planning"
    )
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Path Tracking Comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "curve_integrated_path.png",
        dpi=200
    )

    # Figure 2: Speed
    plt.figure(figsize=(12, 6))
    plt.plot(
        fixed["time"], fixed["speed"],
        label="Fixed Speed"
    )
    plt.plot(
        planned["time"], planned["speed"],
        label="Planned Actual Speed"
    )
    plt.plot(
        planned["time"][:-1],
        planned["target_speed"],
        "--",
        label="Planned Target Speed"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Control Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "curve_integrated_speed.png",
        dpi=200
    )

    # Figure 3: Lateral acceleration
    plt.figure(figsize=(12, 6))
    plt.plot(
        fixed["time"][:-1],
        np.abs(fixed["lateral_acceleration"]),
        label="Fixed Speed"
    )
    plt.plot(
        planned["time"][:-1],
        np.abs(planned["lateral_acceleration"]),
        label="Speed Planning"
    )
    plt.axhline(
        MAX_LATERAL_ACCELERATION,
        color="red",
        linestyle="--",
        label="Planning Limit"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Lateral Acceleration (m/s^2)")
    plt.title("Lateral Acceleration Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "curve_integrated_lateral.png",
        dpi=200
    )

    # Figure 4: Path error
    plt.figure(figsize=(12, 6))
    plt.plot(
        fixed["time"][:-1],
        fixed["path_error"],
        label="Fixed Speed"
    )
    plt.plot(
        planned["time"][:-1],
        planned["path_error"],
        label="Speed Planning"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Path Error (m)")
    plt.title("Path Tracking Error Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "curve_integrated_error.png",
        dpi=200
    )

    # Figure 5: Speed vs distance
    plt.figure(figsize=(12, 6))
    plt.plot(
        PATH_DISTANCE,
        PLANNED_SPEED,
        label="Planned Speed Profile"
    )
    plt.scatter(
        planned["distance"][::10],
        planned["speed"][:-1:10],
        color="red",
        s=15,
        label="Actual Speed"
    )
    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Planning Execution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "curve_integrated_profile.png",
        dpi=200
    )

    print("\nFive figures saved to:", RESULTS_DIR)

    plt.show()


if __name__ == "__main__":
    main()
