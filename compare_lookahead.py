import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController


def run_simulation(lookahead): #giving a value to the lookahead
    vehicle = BicycleModel(wheelbase=2.7)
    controller = PurePursuitController(wheelbase=2.7)

    vehicle.x = 0.0 #original value of all positions
    vehicle.y = -2.0
    vehicle.yaw = 0.0
    vehicle.velocity = 5.0

    path_x = np.linspace(0, 100, 501)
    path_y = 2 * np.sin(0.05 * path_x)

    dt = 0.1
    steps = 200

    history_x = [vehicle.x]
    history_y = [vehicle.y]
    errors = []
    times = []

    for step in range(steps):
        distances = np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )

        nearest_index = int(np.argmin(distances))
        target_index = nearest_index

        while (
            target_index < len(path_x) - 1
            and math.hypot(
                path_x[target_index] - vehicle.x,
                path_y[target_index] - vehicle.y
            ) < lookahead
        ):
            target_index += 1

        steering = controller.calculate_steering(
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            float(path_x[target_index]),
            float(path_y[target_index])
        )

        vehicle.update(
            acceleration=0.0,
            steering_angle=steering,
            dt=dt
        )

        history_x.append(vehicle.x)
        history_y.append(vehicle.y)

        error = float(np.min(np.hypot(
            path_x - vehicle.x,
            path_y - vehicle.y
        )))

        errors.append(error)
        times.append((step + 1) * dt)

    rmse = math.sqrt(np.mean(np.square(errors)))

    return {
        "lookahead": lookahead,
        "x": history_x,
        "y": history_y,
        "times": times,
        "errors": errors,
        "rmse": rmse,
        "max_error": max(errors)
    } #giving results back to the main function 
  


def main(): #this is main function which can hold some data here
    lookahead_values = [2.0, 5.0, 10.0]
    results = [] # the results will br stored in this list

    for lookahead in lookahead_values:
        result = run_simulation(lookahead)
        results.append(result)

        print(
            f"Lookahead = {lookahead:.1f} m | "
            f"RMSE = {result['rmse']:.3f} m | "
            f"Max Error = {result['max_error']:.3f} m"
        )

    
    path_x = np.linspace(0, 100, 501)
    path_y = 2 * np.sin(0.05 * path_x)

    plt.figure(figsize=(11, 6))
    plt.plot(path_x, path_y, "k--", label="Reference Path")

    for result in results:
        plt.plot(
            result["x"],
            result["y"],
            label=f"Lookahead = {result['lookahead']} m"
        )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Lookahead Comparison")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "lookahead_comparison.png", dpi=200)

    
    plt.figure(figsize=(11, 6))

    for result in results:
        plt.plot(
            result["times"],
            result["errors"],
            label=f"Lookahead = {result['lookahead']} m"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Nearest-Path Error (m)")
    plt.title("Tracking Error Comparison")
    plt.grid(True)
    plt.legend()

    plt.savefig(
        results_dir / "tracking_error_comparison.png",
        dpi=200
    )

    plt.show()



if __name__ == "__main__":
    main()