from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.steering_rate_limiter import SteeringRateLimiter
from src.controllers.pid_speed import PIDSpeedController


def main():

    # ==========================================
    # 1. Simulation settings
    # ==========================================

    dt = 0.1
    steps = 300
    target_speed = 10.0

    wheelbase = 2.7

    # Reference path
    path_x = np.linspace(0, 250, 1251)
    path_y = 2 * np.sin(0.05 * path_x)

    # ==========================================
    # 2. Initialize vehicle and controllers
    # ==========================================

    vehicle = BicycleModel(wheelbase=wheelbase)

    vehicle.x = 0.0
    vehicle.y = -2.0
    vehicle.yaw = 0.0
    vehicle.velocity = 0.0

    steering_controller = PurePursuitController(
        wheelbase=wheelbase
    )

    steering_limiter = SteeringRateLimiter(
        max_rate_deg=60.0
    )

    speed_controller = PIDSpeedController(
        kp=1.0,
        ki=0.1,
        kd=0.05,
        min_acceleration=-3.0,
        max_acceleration=2.0,
        anti_windup=True
    )

    # ==========================================
    # 3. Data recording
    # ==========================================

    times = [0.0]
    history_x = [vehicle.x]
    history_y = [vehicle.y]
    history_speed = [vehicle.velocity]

    history_steering = []
    history_acceleration = []
    history_error = []

    # ==========================================
    # 4. Simulation loop
    # ==========================================

    for step in range(steps):

        # Find nearest point on reference path
        distances = np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )

        nearest_index = int(np.argmin(distances))

        # Adaptive lookahead
        lookahead = (
            steering_controller.calculate_lookahead(
                vehicle.velocity
            )
        )

        # Find lookahead target
        target_index = nearest_index

        while target_index < len(path_x) - 1:

            distance_to_target = np.hypot(
                path_x[target_index] - vehicle.x,
                path_y[target_index] - vehicle.y
            )

            if distance_to_target >= lookahead:
                break

            target_index += 1

        target_x = path_x[target_index]
        target_y = path_y[target_index]

        # Lateral control: Pure Pursuit
        desired_steering = (
            steering_controller.calculate_steering(
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                target_x,
                target_y
            )
        )

        # Limit steering rate
        steering = steering_limiter.update(
            desired_steering,
            dt
        )

        # Longitudinal control: PID
        acceleration = speed_controller.update(
            target_speed=target_speed,
            current_speed=vehicle.velocity,
            dt=dt
        )

        # Record lateral tracking error
        tracking_error = distances[nearest_index]

        # Update vehicle
        vehicle.update(
            acceleration=acceleration,
            steering_angle=steering,
            dt=dt
        )

        # Record data
        times.append((step + 1) * dt)

        history_x.append(vehicle.x)
        history_y.append(vehicle.y)
        history_speed.append(vehicle.velocity)

        history_steering.append(
            np.degrees(steering)
        )

        history_acceleration.append(acceleration)
        history_error.append(tracking_error)

    # ==========================================
    # 5. Convert data to NumPy arrays
    # ==========================================

    times = np.array(times)
    history_x = np.array(history_x)
    history_y = np.array(history_y)
    history_speed = np.array(history_speed)

    history_steering = np.array(history_steering)
    history_acceleration = np.array(history_acceleration)
    history_error = np.array(history_error)

    # ==========================================
    # 6. Performance metrics
    # ==========================================

    path_rmse = np.sqrt(
        np.mean(history_error ** 2)
    )

    max_path_error = np.max(history_error)

    final_speed_error = abs(
        target_speed - history_speed[-1]
    )

    print("\n=== Integrated Control Results ===")
    print(f"Path RMSE: {path_rmse:.3f} m")
    print(f"Max Path Error: {max_path_error:.3f} m")
    print(f"Final Speed: {history_speed[-1]:.3f} m/s")
    print(f"Final Speed Error: {final_speed_error:.3f} m/s")

    # ==========================================
    # 7. Save figures
    # ==========================================

    results_dir = (
        Path(__file__).resolve().parent / "results"
    )

    results_dir.mkdir(exist_ok=True)

    # Figure 1: Path tracking
    plt.figure(figsize=(11, 6))

    plt.plot(
        path_x,
        path_y,
        "--",
        label="Reference Path"
    )

    plt.plot(
        history_x,
        history_y,
        label="Vehicle Trajectory"
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Integrated Control: Path Tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_path.png",
        dpi=200
    )

    # Figure 2: Speed control
    plt.figure(figsize=(11, 6))

    plt.plot(
        times,
        history_speed,
        label="Actual Speed"
    )

    plt.axhline(
        target_speed,
        linestyle="--",
        color="black",
        label="Target Speed"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Integrated Control: Speed Response")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_speed.png",
        dpi=200
    )

    # Figure 3: Steering angle
    plt.figure(figsize=(11, 6))

    plt.plot(
        times[:-1],
        history_steering
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (deg)")
    plt.title("Integrated Control: Steering Angle")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_steering.png",
        dpi=200
    )

    # Figure 4: Acceleration
    plt.figure(figsize=(11, 6))

    plt.plot(
        times[:-1],
        history_acceleration
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Integrated Control: Acceleration")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_acceleration.png",
        dpi=200
    )

    # Figure 5: Path tracking error
    plt.figure(figsize=(11, 6))

    plt.plot(
        times[:-1],
        history_error
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title("Integrated Control: Path Error")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_error.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()