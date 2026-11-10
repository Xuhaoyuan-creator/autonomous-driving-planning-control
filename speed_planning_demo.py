from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.planning.speed_planner import SpeedPlanner


def main():

    # Create reference path
    path_x = np.linspace(0, 250, 1251)
    path_y = 2 * np.sin(0.05 * path_x)

    # Initialize speed planner
    planner = SpeedPlanner(
        max_speed=10.0,
        max_lateral_acceleration=0.5,
        max_braking=2.0
    )

    # Generate speed profile
    result = planner.plan(path_x, path_y)

    curvature = result["curvature"]
    distance = result["distance"]
    curvature_speed = result["curvature_speed"]
    speed_profile = result["speed_profile"]

    print("\n=== Speed Planning Results ===")
    print(f"Maximum Speed: {np.max(speed_profile):.3f} m/s")
    print(f"Minimum Speed: {np.min(speed_profile):.3f} m/s")
    print(f"Maximum Curvature: {np.max(curvature):.5f} 1/m")

    results_dir = (
        Path(__file__).resolve().parent / "results"
    )
    results_dir.mkdir(exist_ok=True)

    # Figure 1: Reference path
    plt.figure(figsize=(11, 5))
    plt.plot(path_x, path_y)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Reference Path")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(
        results_dir / "speed_planning_path.png",
        dpi=200
    )

    # Figure 2: Curvature
    plt.figure(figsize=(11, 5))
    plt.plot(distance, curvature)
    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Curvature (1/m)")
    plt.title("Path Curvature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        results_dir / "speed_planning_curvature.png",
        dpi=200
    )

    # Figure 3: Speed profile
    plt.figure(figsize=(11, 5))

    plt.plot(
        distance,
        curvature_speed,
        "--",
        label="Curvature Speed Limit"
    )

    plt.plot(
        distance,
        speed_profile,
        label="Planned Speed"
    )

    plt.axhline(
        planner.max_speed,
        color="black",
        linestyle=":",
        label="Maximum Speed"
    )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Speed (m/s)")
    plt.title("Curvature-Based Speed Planning")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "speed_planning_profile.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()
