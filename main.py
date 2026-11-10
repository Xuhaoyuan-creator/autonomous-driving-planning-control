import math
from pathlib import Path

import matplotlib.pyplot as plt

from src.vehicle_model.bicycle_model import BicycleModel


def main():
    vehicle = BicycleModel(wheelbase=2.7)

    # Simulation parameters
    dt = 0.1
    simulation_time = 10.0
    simulation_steps = int(simulation_time / dt)

    acceleration = 2.0
    steering_angle = math.radians(5) #this step makes the plot look like a curve
    

    # Data storage
    times = [0.0]
    x_positions = [vehicle.x]
    y_positions = [vehicle.y]
    velocities = [vehicle.velocity]
    yaw_angles = [math.degrees(vehicle.yaw)]

    # Simulation loop, keeping recording every step for the car
    for step in range(simulation_steps):
        vehicle.update(acceleration, steering_angle, dt)

        state = vehicle.get_state()

        times.append((step + 1) * dt)
        x_positions.append(state["x"])
        y_positions.append(state["y"])
        velocities.append(state["velocity"])
        yaw_angles.append(math.degrees(state["yaw"]))

    # Create results folder
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Plot 1: Vehicle trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(x_positions, y_positions, label="Vehicle Path")
    plt.scatter(x_positions[0], y_positions[0], label="Start")
    plt.scatter(x_positions[-1], y_positions[-1], label="End")

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Vehicle Trajectory")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "trajectory.png", dpi=200)

    # Plot 2: Velocity
    plt.figure(figsize=(8, 5))
    plt.plot(times, velocities)

    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "velocity.png", dpi=200)

    # Plot 3: Yaw angle
    plt.figure(figsize=(8, 5))
    plt.plot(times, yaw_angles)

    plt.xlabel("Time (s)")
    plt.ylabel("Yaw Angle (deg)")
    plt.title("Yaw Angle vs Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "yaw.png", dpi=200)

    # Final results
    print("Simulation completed!")
    print(f"Final X: {vehicle.x:.2f} m")
    print(f"Final Y: {vehicle.y:.2f} m")
    print(f"Final Velocity: {vehicle.velocity:.2f} m/s")
    print(f"Final Yaw: {math.degrees(vehicle.yaw):.2f} deg")
    print(f"Results saved to: {results_dir}")

    plt.show()


if __name__ == "__main__":
    main()