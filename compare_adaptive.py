
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.steering_rate_limiter import SteeringRateLimiter


def run_simulation(mode, use_rate_limiter=False):

    # 1. Create vehicle and controllers
    vehicle = BicycleModel(wheelbase=2.7)
    controller = PurePursuitController(wheelbase=2.7)
    limiter = SteeringRateLimiter(max_rate_deg=60)

    # 2. Initial vehicle state
    vehicle.x = 0.0
    vehicle.y = -2.0
    vehicle.yaw = 0.0
    vehicle.velocity = 2.0

    # 3. Reference path
    path_x = np.linspace(0, 200, 1001)
    path_y = 2 * np.sin(0.05 * path_x)

    # 4. Simulation parameters
    dt = 0.1
    steps = 200
    acceleration = 0.5

    # 5. Data recording
    history_x = [vehicle.x]
    history_y = [vehicle.y]
    history_time = [0.0]
    history_velocity = [vehicle.velocity]

    history_lookahead = []
    history_error = []
    history_steering = []

    # 6. Main simulation loop
    for step in range(steps):

        # Choose lookahead distance
        if mode == "fixed":
            lookahead = 5.0

        elif mode == "adaptive":
            lookahead = controller.calculate_lookahead(
                vehicle.velocity
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Find nearest path point
        distances = np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )

        nearest_index = int(np.argmin(distances))
        target_index = nearest_index

        # Find lookahead target point
        while (
            target_index < len(path_x) - 1
            and math.hypot(
                path_x[target_index] - vehicle.x,
                path_y[target_index] - vehicle.y
            ) < lookahead
        ):
            target_index += 1

        target_x = float(path_x[target_index])
        target_y = float(path_y[target_index])

        # Calculate desired steering angle
        desired_steering = controller.calculate_steering(
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            target_x,
            target_y
        )

        # Apply steering rate limiter if enabled
        if use_rate_limiter:
            steering = limiter.update(
                desired_steering,
                dt
            )
        else:
            steering = desired_steering

        # Record actual steering angle in degrees
        history_steering.append(math.degrees(steering))

        # Update vehicle state
        vehicle.update(
            acceleration=acceleration,
            steering_angle=steering,
            dt=dt
        )

        # Record vehicle states
        history_x.append(vehicle.x)
        history_y.append(vehicle.y)
        history_time.append((step + 1) * dt)
        history_velocity.append(vehicle.velocity)
        history_lookahead.append(lookahead)

        # Calculate nearest-path tracking error
        error = float(
            np.min(
                np.hypot(
                    path_x - vehicle.x,
                    path_y - vehicle.y
                )
            )
        )

        history_error.append(error)

    # 7. Calculate performance metrics
    rmse = math.sqrt(
        np.mean(np.square(history_error))
    )

    # 8. Return simulation results
    return {
        "mode": mode,
        "use_rate_limiter": use_rate_limiter,
        "x": history_x,
        "y": history_y,
        "time": history_time,
        "velocity": history_velocity,
        "lookahead": history_lookahead,
        "error": history_error,
        "steering": history_steering,
        "rmse": rmse,
        "max_error": max(history_error)
    }


def main():

    # Compare fixed and adaptive lookahead
    # Both runs use the original steering behavior
    fixed = run_simulation(
        "fixed",
        use_rate_limiter=False
    )

    adaptive = run_simulation(
        "adaptive",
        use_rate_limiter=False
    )

    # Print tracking accuracy
    print("Fixed Lookahead")
    print(f"RMSE: {fixed['rmse']:.3f} m")
    print(f"Max Error: {fixed['max_error']:.3f} m")

    print("\nAdaptive Lookahead")
    print(f"RMSE: {adaptive['rmse']:.3f} m")
    print(f"Max Error: {adaptive['max_error']:.3f} m")

    # Create results folder
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Reference path
    path_x = np.linspace(0, 200, 1001)
    path_y = 2 * np.sin(0.05 * path_x)

    # ---------------------------------
    # Figure 1: Vehicle trajectories
    # ---------------------------------
    plt.figure(figsize=(11, 6))

    plt.plot(
        path_x,
        path_y,
        "k--",
        label="Reference Path"
    )

    plt.plot(
        fixed["x"],
        fixed["y"],
        label="Fixed Lookahead"
    )

    plt.plot(
        adaptive["x"],
        adaptive["y"],
        label="Adaptive Lookahead"
    )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Fixed vs Adaptive Pure Pursuit")
    plt.xlim(0, 150)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "adaptive_trajectory.png",
        dpi=200
    )

    # ---------------------------------
    # Figure 2: Tracking errors
    # ---------------------------------
    plt.figure(figsize=(11, 6))

    plt.plot(
        fixed["time"][1:],
        fixed["error"],
        label="Fixed Lookahead"
    )

    plt.plot(
        adaptive["time"][1:],
        adaptive["error"],
        label="Adaptive Lookahead"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Nearest-Path Error (m)")
    plt.title("Tracking Error Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "adaptive_error.png",
        dpi=200
    )

    # ---------------------------------
    # Figure 3: Lookahead distance
    # ---------------------------------
    plt.figure(figsize=(11, 6))

    plt.plot(
        fixed["time"][:-1],
        fixed["lookahead"],
        label="Fixed Lookahead"
    )

    plt.plot(
        adaptive["time"][:-1],
        adaptive["lookahead"],
        label="Adaptive Lookahead"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Lookahead Distance (m)")
    plt.title("Lookahead Distance vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "adaptive_lookahead.png",
        dpi=200
    )

    # ---------------------------------
    # Figure 4: Steering angles
    # ---------------------------------
    plt.figure(figsize=(11, 6))

    plt.plot(
        fixed["time"][:-1],
        fixed["steering"],
        label="Fixed Lookahead"
    )

    plt.plot(
        adaptive["time"][:-1],
        adaptive["steering"],
        label="Adaptive Lookahead"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (deg)")
    plt.title("Steering Angle Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "steering_comparison.png",
        dpi=200
    )

    # ---------------------------------
    # Steering smoothness analysis
    # ---------------------------------
    fixed_changes = np.diff(fixed["steering"])
    adaptive_changes = np.diff(adaptive["steering"])

    print("\nSteering Smoothness Analysis")

    print(
        "Fixed mean steering change:",
        f"{np.mean(np.abs(fixed_changes)):.3f}",
        "deg/step"
    )

    print(
        "Adaptive mean steering change:",
        f"{np.mean(np.abs(adaptive_changes)):.3f}",
        "deg/step"
    )

    # Display all figures
    plt.show()


if __name__ == "__main__":
    main()
