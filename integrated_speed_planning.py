
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
STEPS = 350
WHEELBASE = 2.7

MAX_SPEED = 10.0
MAX_ACCELERATION = 2.0
MAX_BRAKING = 3.0
MAX_LATERAL_ACCELERATION = 0.5

# Stop simulation near the path endpoint.
# This is not a physical stopping controller.
END_DISTANCE_THRESHOLD = 2.0

VERSION = "INTEGRATED_SPEED_PLANNING_V5"


def main():

    print("\n" + "=" * 55)
    print(VERSION)
    print("=" * 55)
    print("Running file:", Path(__file__).resolve())

    # ==========================================
    # 2. REFERENCE PATH
    # ==========================================

    path_x = np.linspace(0.0, 250.0, 1251)
    path_y = 2.0 * np.sin(0.05 * path_x)

    # ==========================================
    # 3. SPEED PLANNER
    # ==========================================

    planner = SpeedPlanner(
        max_speed=MAX_SPEED,
        max_lateral_acceleration=MAX_LATERAL_ACCELERATION,
        max_braking=MAX_BRAKING,
        max_acceleration=MAX_ACCELERATION
    )

    # Do not force the starting target speed to zero.
    # The vehicle itself still starts from rest.
    plan = planner.plan(path_x, path_y)

    planned_speed = np.asarray(
        plan["speed_profile"],
        dtype=float
    )

    path_distance = np.asarray(
        plan["distance"],
        dtype=float
    )

    if len(planned_speed) != len(path_x):
        raise RuntimeError(
            "Speed profile length does not match path."
        )

    if not np.all(np.isfinite(planned_speed)):
        raise RuntimeError(
            "Speed profile contains invalid values."
        )

    if planned_speed[0] <= 0:
        raise RuntimeError(
            "Starting target speed is zero."
        )

    print("\n=== SPEED PLANNER CHECK ===")
    print("First 10 speeds:", planned_speed[:10])
    print("Minimum planned speed:", planned_speed.min())
    print("Maximum planned speed:", planned_speed.max())

    # ==========================================
    # 4. VEHICLE
    # ==========================================

    vehicle = BicycleModel(wheelbase=WHEELBASE)

    vehicle.x = 0.0
    vehicle.y = -2.0
    vehicle.yaw = 0.0
    vehicle.velocity = 0.0

    # ==========================================
    # 5. CONTROLLERS
    # ==========================================

    lateral_controller = PurePursuitController(
        wheelbase=WHEELBASE
    )

    steering_limiter = SteeringRateLimiter(
        max_rate_deg=60.0
    )

    speed_controller = PIDSpeedController(
        kp=1.0,
        ki=0.1,
        kd=0.05,
        min_acceleration=-MAX_BRAKING,
        max_acceleration=MAX_ACCELERATION,
        anti_windup=True
    )

    # ==========================================
    # 6. DATA STORAGE
    # ==========================================

    time_history = [0.0]

    x_history = [vehicle.x]
    y_history = [vehicle.y]
    speed_history = [vehicle.velocity]

    target_speed_history = []
    acceleration_history = []
    steering_history = []
    path_error_history = []
    distance_history = []
    lateral_acceleration_history = []

    stop_reason = "Maximum simulation steps reached"

    # ==========================================
    # 7. SIMULATION LOOP
    # ==========================================

    print("\n=== SIMULATION DEBUG ===")

    for step in range(STEPS):

        # A. Find nearest path point
        distances = np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )

        nearest_index = int(np.argmin(distances))

        current_s = float(
            path_distance[nearest_index]
        )

        # B. Stop near endpoint
        endpoint_distance = np.hypot(
            vehicle.x - path_x[-1],
            vehicle.y - path_y[-1]
        )

        if endpoint_distance <= END_DISTANCE_THRESHOLD:
            stop_reason = "Reached path endpoint"
            print(
                f"Endpoint reached at "
                f"t={step * DT:.2f} s"
            )
            break

        # C. Target speed
        target_speed = float(
            planned_speed[nearest_index]
        )

        # D. Adaptive lookahead
        lookahead = (
            lateral_controller.calculate_lookahead(
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

        target_x = path_x[target_index]
        target_y = path_y[target_index]

        # E. Lateral control
        desired_steering = (
            lateral_controller.calculate_steering(
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                target_x,
                target_y
            )
        )

        steering_angle = steering_limiter.update(
            desired_steering,
            DT
        )

        # F. Longitudinal control
        acceleration = speed_controller.update(
            target_speed=target_speed,
            current_speed=vehicle.velocity,
            dt=DT
        )

        if not np.isfinite(acceleration):
            raise RuntimeError(
                f"Invalid acceleration at step {step}"
            )

        # G. Lateral acceleration estimate
        lateral_acceleration = (
            vehicle.velocity ** 2
            / WHEELBASE
            * np.tan(steering_angle)
        )

        # H. Debug output
        if step < 10 or step % 20 == 0:
            print(
                f"Step {step:3d} | "
                f"Time {step * DT:5.1f}s | "
                f"Index {nearest_index:4d} | "
                f"Target {target_speed:6.3f} | "
                f"Actual {vehicle.velocity:6.3f} | "
                f"Accel {acceleration:6.3f}"
            )

        # I. Record control values
        target_speed_history.append(target_speed)

        acceleration_history.append(
            acceleration
        )

        steering_history.append(
            np.degrees(steering_angle)
        )

        path_error_history.append(
            distances[nearest_index]
        )

        distance_history.append(current_s)

        lateral_acceleration_history.append(
            lateral_acceleration
        )

        # J. Update vehicle
        vehicle.update(
            acceleration=acceleration,
            steering_angle=steering_angle,
            dt=DT
        )

        if not np.all(
            np.isfinite([
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                vehicle.velocity
            ])
        ):
            raise RuntimeError(
                f"Invalid vehicle state at step {step}"
            )

        # K. Record updated state
        time_history.append(
            (step + 1) * DT
        )

        x_history.append(vehicle.x)
        y_history.append(vehicle.y)
        speed_history.append(vehicle.velocity)

    # ==========================================
    # 8. CONVERT ARRAYS
    # ==========================================

    time_history = np.asarray(time_history)

    x_history = np.asarray(x_history)
    y_history = np.asarray(y_history)
    speed_history = np.asarray(speed_history)

    target_speed_history = np.asarray(
        target_speed_history
    )

    acceleration_history = np.asarray(
        acceleration_history
    )

    steering_history = np.asarray(
        steering_history
    )

    path_error_history = np.asarray(
        path_error_history
    )

    distance_history = np.asarray(
        distance_history
    )

    lateral_acceleration_history = np.asarray(
        lateral_acceleration_history
    )

    if len(target_speed_history) == 0:
        raise RuntimeError(
            "No simulation steps were executed."
        )

    # ==========================================
    # 9. PERFORMANCE METRICS
    # ==========================================

    path_rmse = np.sqrt(
        np.mean(path_error_history ** 2)
    )

    max_path_error = np.max(
        path_error_history
    )

    speed_error = (
        target_speed_history
        - speed_history[:-1]
    )

    speed_rmse = np.sqrt(
        np.mean(speed_error ** 2)
    )

    max_actual_lateral_acceleration = np.max(
        np.abs(lateral_acceleration_history)
    )

    total_distance = np.sum(
        np.hypot(
            np.diff(x_history),
            np.diff(y_history)
        )
    )

    print("\n" + "=" * 55)
    print("SIMULATION RESULTS")
    print("=" * 55)

    print("Stop reason:", stop_reason)
    print(f"Executed steps: {len(target_speed_history)}")

    print(f"Path RMSE: {path_rmse:.3f} m")
    print(f"Maximum Path Error: {max_path_error:.3f} m")

    print(f"Speed RMSE: {speed_rmse:.3f} m/s")

    print(
        f"Final Speed: {speed_history[-1]:.3f} m/s"
    )

    print(
        f"Maximum Actual Speed: "
        f"{speed_history.max():.3f} m/s"
    )

    # ASCII-only terminal output avoids GBK errors.
    print(
        f"Maximum Lateral Acceleration: "
        f"{max_actual_lateral_acceleration:.3f} m/s^2"
    )

    print(
        f"Total Distance Traveled: "
        f"{total_distance:.3f} m"
    )

    print(
        f"Final Position: "
        f"({x_history[-1]:.3f}, {y_history[-1]:.3f})"
    )

    if (
        max_actual_lateral_acceleration
        > MAX_LATERAL_ACCELERATION
    ):
        print(
            "WARNING: Actual lateral acceleration "
            "exceeded the planning limit."
        )

    # ==========================================
    # 10. RESULTS DIRECTORY
    # ==========================================

    results_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================
    # FIGURE 1: PATH TRACKING
    # ==========================================

    fig, ax = plt.subplots(figsize=(12, 6))

    x_min = max(
        float(path_x.min()),
        float(x_history.min()) - 5.0
    )

    x_max = min(
        float(path_x.max()),
        float(x_history.max()) + 5.0
    )

    visible = (
        (path_x >= x_min)
        & (path_x <= x_max)
    )

    ax.plot(
        path_x[visible],
        path_y[visible],
        "k--",
        linewidth=2,
        label="Reference Path",
        zorder=1
    )

    ax.plot(
        x_history,
        y_history,
        color="red",
        linewidth=2.5,
        label="Actual Vehicle Trajectory",
        zorder=3
    )

    ax.scatter(
        x_history[0],
        y_history[0],
        color="green",
        s=100,
        label="Start",
        zorder=5
    )

    ax.scatter(
        x_history[-1],
        y_history[-1],
        color="blue",
        marker="x",
        s=100,
        label="End",
        zorder=5
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(
        "Reference Path vs Actual Vehicle Trajectory"
    )

    ax.set_xlim(
        min(x_min, float(x_history.min()) - 1),
        max(x_max, float(x_history.max()) + 1)
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        results_dir / "planned_path_tracking.png",
        dpi=200
    )

    # ==========================================
    # FIGURE 2: PLANNED VS ACTUAL SPEED
    # ==========================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        time_history[:-1],
        target_speed_history,
        "--",
        linewidth=2,
        label="Planned Target Speed"
    )

    plt.plot(
        time_history,
        speed_history,
        linewidth=2,
        label="Actual Speed"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Planned vs Actual Speed")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "planned_vs_actual_speed.png",
        dpi=200
    )

    # ==========================================
    # FIGURE 3: ACCELERATION
    # ==========================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        time_history[:-1],
        acceleration_history,
        label="Acceleration"
    )

    plt.axhline(
        MAX_ACCELERATION,
        color="red",
        linestyle="--",
        label="Maximum Acceleration"
    )

    plt.axhline(
        -MAX_BRAKING,
        color="orange",
        linestyle="--",
        label="Maximum Braking"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.title("Acceleration Control")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "planned_acceleration.png",
        dpi=200
    )

    # ==========================================
    # FIGURE 4: STEERING
    # ==========================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        time_history[:-1],
        steering_history
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (deg)")
    plt.title("Steering Control")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "planned_steering.png",
        dpi=200
    )

    # ==========================================
    # FIGURE 5: PATH ERROR
    # ==========================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        time_history[:-1],
        path_error_history
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Path Error (m)")
    plt.title("Path Tracking Error")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "planned_path_error.png",
        dpi=200
    )

    # ==========================================
    # FIGURE 6: SPEED VS DISTANCE
    # ==========================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        path_distance,
        planned_speed,
        linewidth=2,
        label="Planned Speed Profile"
    )

    plt.scatter(
        distance_history[::10],
        speed_history[:-1:10],
        color="red",
        s=18,
        label="Actual Vehicle Speed"
    )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("Speed Profile Along Path")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "planned_speed_distance.png",
        dpi=200
    )

    print("\nAll six figures saved to:", results_dir)

    plt.show()


if __name__ == "__main__":
    main()
