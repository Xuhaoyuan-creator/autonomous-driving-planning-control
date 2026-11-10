from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pid_speed import PIDSpeedController


def run_simulation(kp, ki=0.1, kd=0.05):

    # Create vehicle and PID controller
    vehicle = BicycleModel(wheelbase=2.7)

    controller = PIDSpeedController(
        kp=kp,
        ki=ki,
        kd=kd,
        min_acceleration=-3.0,
        max_acceleration=2.0
    )

    # Initial conditions
    vehicle.velocity = 0.0

    target_speed = 10.0
    dt = 0.1
    steps = 200

    # Record data
    times = [0.0]
    speeds = [vehicle.velocity]
    accelerations = []

    # Simulation loop
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

    # Convert to numpy arrays
    times = np.array(times)
    speeds = np.array(speeds)
    accelerations = np.array(accelerations)

    # Performance metrics
    speed_error = target_speed - speeds

    rmse = np.sqrt(np.mean(speed_error**2))

    max_speed = np.max(speeds)

    overshoot = max(
        0.0,
        (max_speed - target_speed) / target_speed * 100
    )

    final_error = abs(target_speed - speeds[-1])

    # Settling time:
    # First time after which speed stays within 2%
    tolerance = 0.02 * target_speed
    settling_time = None

    for i in range(len(speeds)):
        if np.all(
            np.abs(speed_error[i:]) <= tolerance
        ):
            settling_time = times[i]
            break

    return {
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "time": times,
        "speed": speeds,
        "acceleration": accelerations,
        "target_speed": target_speed,
        "rmse": rmse,
        "overshoot": overshoot,
        "final_error": final_error,
        "settling_time": settling_time
    }


def main():

    # Compare three Kp values
    kp_values = [0.4, 1.0, 2.0]

    results = []

    for kp in kp_values:
        result = run_simulation(kp=kp)
        results.append(result)

    # Print metrics
    print("\n=== PID Parameter Comparison ===")

    for result in results:

        print(f"\nKp = {result['kp']}")

        print(
            f"Speed RMSE: "
            f"{result['rmse']:.3f} m/s"
        )

        print(
            f"Overshoot: "
            f"{result['overshoot']:.2f}%"
        )

        print(
            f"Final Error: "
            f"{result['final_error']:.3f} m/s"
        )

        if result["settling_time"] is None:
            print("Settling Time: Not settled")
        else:
            print(
                f"Settling Time: "
                f"{result['settling_time']:.2f} s"
            )

    # Create output directory
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Figure 1: Speed response comparison
    plt.figure(figsize=(11, 6))

    for result in results:
        plt.plot(
            result["time"],
            result["speed"],
            label=f"Kp = {result['kp']}"
        )

    plt.axhline(
        10.0,
        color="black",
        linestyle="--",
        label="Target Speed"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("PID Speed Response Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "pid_kp_speed.png",
        dpi=200
    )

    # Figure 2: Acceleration comparison
    plt.figure(figsize=(11, 6))

    for result in results:
        plt.plot(
            result["time"][:-1],
            result["acceleration"],
            label=f"Kp = {result['kp']}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("PID Acceleration Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "pid_kp_acceleration.png",
        dpi=200
    )

    # Figure 3: Absolute speed error
    plt.figure(figsize=(11, 6))

    for result in results:

        error = np.abs(
            result["target_speed"] - result["speed"]
        )

        plt.plot(
            result["time"],
            error,
            label=f"Kp = {result['kp']}"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Absolute Speed Error (m/s)")
    plt.title("PID Speed Error Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "pid_kp_error.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()
