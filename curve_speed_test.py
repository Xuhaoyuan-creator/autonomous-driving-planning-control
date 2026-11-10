
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.planning.speed_planner import SpeedPlanner


def main():

    # ======================================
    # 1. Experiment settings
    # ======================================

    max_speed = 10.0
    max_lateral_acceleration = 0.5
    max_acceleration = 2.0
    max_braking = 3.0

    # ======================================
    # 2. Create two reference paths
    # ======================================

    path_x = np.linspace(0, 250, 1251)

    # Previous road
    old_path_y = 2 * np.sin(0.05 * path_x)

    # New road with sharper curves
    new_path_y = 5 * np.sin(0.08 * path_x)

    # ======================================
    # 3. Initialize speed planner
    # ======================================

    planner = SpeedPlanner(
        max_speed=max_speed,
        max_lateral_acceleration=max_lateral_acceleration,
        max_acceleration=max_acceleration,
        max_braking=max_braking
    )

    # ======================================
    # 4. Plan speeds
    # ======================================

    old_result = planner.plan(
        path_x,
        old_path_y
    )

    new_result = planner.plan(
        path_x,
        new_path_y
    )

    old_speed = old_result["speed_profile"]
    new_speed = new_result["speed_profile"]

    old_curvature = old_result["curvature"]
    new_curvature = new_result["curvature"]

    old_s = old_result["distance"]
    new_s = new_result["distance"]

    # ======================================
    # 5. Print experiment results
    # ======================================

    print("\n=== CURVE SPEED TEST ===")

    print("\nOLD PATH")
    print(
        f"Maximum curvature: "
        f"{np.max(old_curvature):.5f} 1/m"
    )
    print(
        f"Minimum planned speed: "
        f"{np.min(old_speed):.3f} m/s"
    )

    print("\nNEW PATH")
    print(
        f"Maximum curvature: "
        f"{np.max(new_curvature):.5f} 1/m"
    )
    print(
        f"Minimum planned speed: "
        f"{np.min(new_speed):.3f} m/s"
    )

    print(
        f"Maximum planned speed: "
        f"{np.max(new_speed):.3f} m/s"
    )

    # Check planned lateral acceleration
    planned_lateral_acceleration = (
        new_speed ** 2 * new_curvature
    )

    print(
        f"Maximum planned lateral acceleration: "
        f"{np.max(planned_lateral_acceleration):.3f} m/s^2"
    )

    # Check speed changes over distance
    ds = np.diff(new_s)
    v_squared_change = np.diff(new_speed ** 2)

    equivalent_acceleration = (
        v_squared_change / (2 * ds)
    )

    print(
        f"Maximum planned acceleration: "
        f"{np.max(equivalent_acceleration):.3f} m/s^2"
    )

    print(
        f"Maximum planned braking: "
        f"{-np.min(equivalent_acceleration):.3f} m/s^2"
    )

    # ======================================
    # 6. Output directory
    # ======================================

    results_dir = (
        Path(__file__).resolve().parent
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================
    # Figure 1: Compare reference paths
    # ======================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        path_x,
        old_path_y,
        "--",
        label="Old Reference Path"
    )

    plt.plot(
        path_x,
        new_path_y,
        label="New Sharper Path"
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Reference Path Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "curve_test_paths.png",
        dpi=200
    )

    # ======================================
    # Figure 2: Compare curvature
    # ======================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        old_s,
        old_curvature,
        label="Old Path Curvature"
    )

    plt.plot(
        new_s,
        new_curvature,
        label="New Path Curvature"
    )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Curvature (1/m)")
    plt.title("Path Curvature Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "curve_test_curvature.png",
        dpi=200
    )

    # ======================================
    # Figure 3: Compare speed profiles
    # ======================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        old_s,
        old_speed,
        "--",
        label="Old Path Planned Speed"
    )

    plt.plot(
        new_s,
        new_speed,
        linewidth=2,
        label="New Path Planned Speed"
    )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Planned Speed (m/s)")
    plt.title("Speed Planning Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "curve_test_speed.png",
        dpi=200
    )

    # ======================================
    # Figure 4: Lateral acceleration
    # ======================================

    plt.figure(figsize=(11, 5))

    plt.plot(
        new_s,
        planned_lateral_acceleration,
        label="Planned Lateral Acceleration"
    )

    plt.axhline(
        max_lateral_acceleration,
        color="red",
        linestyle="--",
        label="Lateral Acceleration Limit"
    )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Lateral Acceleration (m/s^2)")
    plt.title("Planned Lateral Acceleration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "curve_test_lateral_accel.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()
