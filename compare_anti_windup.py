from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pid_speed import PIDSpeedController


def run_simulation(anti_windup):

    vehicle = BicycleModel(wheelbase=2.7)

    controller = PIDSpeedController(
        kp=1.0,
        ki=0.1,
        kd=0.05,
        min_acceleration=-3.0,
        max_acceleration=2.0,
        anti_windup=anti_windup
    )

    vehicle.velocity = 0.0

    dt = 0.1
    steps = 200
    target_speed = 10.0

    times = [0.0]
    speeds = [vehicle.velocity]
    accelerations = []
    integrals = [controller.integral]

    for step in range(steps):

        acceleration = controller.update(
            target_speed=target_speed,
            current_speed=vehicle.velocity,
            dt=dt
        )

        vehicle.update(
            acceleration=acceleration,
            steering_angle=0.0,
            dt=dt
        )

        times.append((step + 1) * dt)
        speeds.append(vehicle.velocity)
        accelerations.append(acceleration)
        integrals.append(controller.integral)

    times = np.array(times)
    speeds = np.array(speeds)
    accelerations = np.array(accelerations)
    integrals = np.array(integrals)

    errors = target_speed - speeds

    rmse = np.sqrt(np.mean(errors**2))
    overshoot = max(
        0.0,
        (np.max(speeds) - target_speed) / target_speed * 100
    )

    return {
        "time": times,
        "speed": speeds,
        "acceleration": accelerations,
        "integral": integrals,
        "rmse": rmse,
        "overshoot": overshoot,
        "final_error": abs(errors[-1])
    }


def main():

    original = run_simulation(anti_windup=False)
    improved = run_simulation(anti_windup=True)

    print("\n=== Anti-Windup Comparison ===")

    print("\nOriginal PID:")
    print(f"RMSE: {original['rmse']:.3f} m/s")
    print(f"Overshoot: {original['overshoot']:.2f}%")
    print(f"Final Error: {original['final_error']:.3f} m/s")

    print("\nPID with Anti-Windup:")
    print(f"RMSE: {improved['rmse']:.3f} m/s")
    print(f"Overshoot: {improved['overshoot']:.2f}%")
    print(f"Final Error: {improved['final_error']:.3f} m/s")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Figure 1: Speed comparison
    plt.figure(figsize=(11, 6))

    plt.plot(
        original["time"],
        original["speed"],
        label="Original PID"
    )

    plt.plot(
        improved["time"],
        improved["speed"],
        label="PID + Anti-Windup"
    )

    plt.axhline(
        10.0,
        color="black",
        linestyle="--",
        label="Target Speed"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("Anti-Windup Speed Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "anti_windup_speed.png",
        dpi=200
    )

    # Figure 2: Acceleration comparison
    plt.figure(figsize=(11, 6))

    plt.plot(
        original["time"][:-1],
        original["acceleration"],
        label="Original PID"
    )

    plt.plot(
        improved["time"][:-1],
        improved["acceleration"],
        label="PID + Anti-Windup"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Anti-Windup Acceleration Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "anti_windup_acceleration.png",
        dpi=200
    )

    # Figure 3: Integral accumulation
    plt.figure(figsize=(11, 6))

    plt.plot(
        original["time"],
        original["integral"],
        label="Original PID"
    )

    plt.plot(
        improved["time"],
        improved["integral"],
        label="PID + Anti-Windup"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Accumulated Speed Error (m)")
    plt.title("Integral Accumulation Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "anti_windup_integral.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()