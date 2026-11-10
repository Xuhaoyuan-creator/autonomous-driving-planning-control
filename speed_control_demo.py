
from pathlib import Path

import matplotlib.pyplot as plt

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pid_speed import PIDSpeedController


def main():
    # Create vehicle
    vehicle = BicycleModel(wheelbase=2.7)

    # Create PID controller
    controller = PIDSpeedController(
        kp=1.0,
        ki=0.1,
        kd=0.05
    )

    # Initial state
    vehicle.velocity = 0.0

    # Target speed
    target_speed = 10.0

    # Simulation settings
    dt = 0.1
    steps = 200

    # Record data
    times = [0.0]
    speeds = [vehicle.velocity]
    accelerations = []

    for step in range(steps):

        # Calculate acceleration using PID
        acceleration = controller.update(
            target_speed=target_speed,
            current_speed=vehicle.velocity,
            dt=dt
        )

        # Update vehicle
        vehicle.update(
            acceleration=acceleration,
            steering_angle=0.0,
            dt=dt
        )

        # Save data
        times.append((step + 1) * dt)
        speeds.append(vehicle.velocity)
        accelerations.append(acceleration)

    # Create results folder
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Figure 1: Speed response
    plt.figure(figsize=(10, 6))

    plt.plot(
        times,
        speeds,
        label="Actual Speed"
    )

    plt.axhline(
        target_speed,
        color="red",
        linestyle="--",
        label="Target Speed"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title("PID Speed Control")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "pid_speed.png",
        dpi=200
    )

    # Figure 2: Acceleration response
    plt.figure(figsize=(10, 6))

    plt.plot(
        times[:-1],
        accelerations,
        label="Acceleration"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("PID Acceleration Command")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "pid_acceleration.png",
        dpi=200
    )

    print(f"Target speed: {target_speed:.2f} m/s")
    print(f"Final speed: {vehicle.velocity:.2f} m/s")
    print(
        f"Final speed error: "
        f"{abs(target_speed - vehicle.velocity):.3f} m/s"
    )

    plt.show()


if __name__ == "__main__":
    main()
