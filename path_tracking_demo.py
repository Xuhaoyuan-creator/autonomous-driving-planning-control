import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController


def main():
    vehicle = BicycleModel(wheelbase=2.7)
    controller = PurePursuitController(wheelbase=2.7)

    # Initial vehicle state
    vehicle.x = 0.0
    vehicle.y = -2.0
    vehicle.yaw = 0.0
    vehicle.velocity = 5.0

    # Reference path
    path_x = np.linspace(0, 100, 501)
    path_y = 2 * np.sin(0.05 * path_x)

    dt = 0.1
    

    history_x = [vehicle.x]
    history_y = [vehicle.y]
    errors = []

    for step in range(200):
        lookahead = controller.calculate_lookahead(vehicle.velocity)
        # Find the nearest point
        distances = np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )

        nearest_index = int(np.argmin(distances))

        # Choose a point ahead along the reference path
        target_index = nearest_index

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

        steering = controller.calculate_steering(
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            target_x,
            target_y
        )

        vehicle.update(
            acceleration=0.0,
            steering_angle=steering,
            dt=dt
        )

        history_x.append(vehicle.x)
        history_y.append(vehicle.y)

        # Approximate nearest-path distance
        error = float(np.min(np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )))
        errors.append(error)

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(path_x, path_y, "--", label="Reference Path")
    plt.plot(history_x, history_y, label="Vehicle Trajectory")

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Pure Pursuit Path Tracking")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    plt.savefig(results_dir / "pure_pursuit_tracking.png", dpi=200)

    print(f"Mean nearest-path distance: {np.mean(errors):.3f} m")
    print(f"Max nearest-path distance: {np.max(errors):.3f} m")

    plt.show()


if __name__ == "__main__":
    main()