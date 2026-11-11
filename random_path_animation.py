
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import (
    FuncAnimation,
    PillowWriter
)

from matplotlib.patches import Polygon

from random_path_tracking import run_simulation


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = (
    ROOT / "results" / "random_tracking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Choose the road to animate.
SCENARIO = "random_sharp"

FPS = 20

# One animation frame for every 3 simulation
# samples. Increase this value to make
# the GIF smaller and faster to render.
FRAME_SKIP = 3

CAR_LENGTH = 4.2
CAR_WIDTH = 1.8


# ==========================================
# VEHICLE GEOMETRY
# ==========================================

def get_vehicle_corners(x, y, yaw):

    corners = np.array([
        [CAR_LENGTH / 2, CAR_WIDTH / 2],
        [CAR_LENGTH / 2, -CAR_WIDTH / 2],
        [-CAR_LENGTH / 2, -CAR_WIDTH / 2],
        [-CAR_LENGTH / 2, CAR_WIDTH / 2],
    ])

    rotation = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw), math.cos(yaw)]
    ])

    return (
        corners @ rotation.T
        + np.array([x, y])
    )


# ==========================================
# CREATE ANIMATION
# ==========================================

def create_animation():

    print("=" * 60)
    print("RANDOM PATH VEHICLE ANIMATION")
    print("=" * 60)

    print("Scenario:", SCENARIO)

    (
        metrics,
        history,
        path_x,
        path_y,
        speed_profile
    ) = run_simulation(SCENARIO)

    x = history["x"]
    y = history["y"]
    yaw = history["yaw"]

    if len(x) < 2:
        raise RuntimeError(
            "Not enough simulation data."
        )

    frame_indices = np.arange(
        0,
        len(x),
        FRAME_SKIP,
        dtype=int
    )

    if frame_indices[-1] != len(x) - 1:
        frame_indices = np.append(
            frame_indices,
            len(x) - 1
        )

    # --------------------------------------
    # FIGURE
    # --------------------------------------

    fig, (ax_path, ax_info) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={
            "width_ratios": [3, 1]
        }
    )

    fig.suptitle(
        "Autonomous Vehicle Random Path Tracking",
        fontsize=15,
        fontweight="bold"
    )

    # --------------------------------------
    # LEFT: PATH TRACKING
    # --------------------------------------

    ax_path.plot(
        path_x,
        path_y,
        "k--",
        linewidth=1.8,
        label="Reference Path"
    )

    trajectory_line, = ax_path.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.5,
        label="Actual Trajectory"
    )

    vehicle = Polygon(
        get_vehicle_corners(
            x[0],
            y[0],
            yaw[0]
        ),
        closed=True,
        facecolor="tab:red",
        edgecolor="black",
        linewidth=1.0,
        alpha=0.9
    )

    ax_path.add_patch(vehicle)

    ax_path.scatter(
        path_x[0],
        path_y[0],
        color="green",
        s=50,
        label="Start"
    )

    ax_path.scatter(
        path_x[-1],
        path_y[-1],
        color="orange",
        s=50,
        label="Destination"
    )

    ax_path.set_xlabel("X Position (m)")
    ax_path.set_ylabel("Y Position (m)")

    ax_path.set_title(
        f"Scenario: {SCENARIO}"
    )

    ax_path.grid(True)
    ax_path.legend(loc="upper right")

    ax_path.set_aspect(
        "equal",
        adjustable="box"
    )

    # --------------------------------------
    # RIGHT: INFORMATION PANEL
    # --------------------------------------

    ax_info.axis("off")

    info_text = ax_info.text(
        0.04,
        0.95,
        "",
        transform=ax_info.transAxes,
        fontsize=11,
        family="monospace",
        verticalalignment="top",
        linespacing=1.7
    )

    # --------------------------------------
    # FRAME UPDATE
    # --------------------------------------

    def update(frame_number):

        index = int(
            frame_indices[frame_number]
        )

        trajectory_line.set_data(
            x[:index + 1],
            y[:index + 1]
        )

        vehicle.set_xy(
            get_vehicle_corners(
                x[index],
                y[index],
                yaw[index]
            )
        )

        # Camera follows the vehicle.
        ax_path.set_xlim(
            x[index] - 22,
            x[index] + 45
        )

        ax_path.set_ylim(
            y[index] - 17,
            y[index] + 17
        )

        time_value = history["time"][index]
        speed = history["speed"][index]
        target_speed = history["target_speed"][index]
        error = history["path_error"][index]
        steering = history["steering_deg"][index]

        lateral_acc = (
            history["lateral_acceleration"][index]
        )

        information = (
            f"Time: {time_value:6.1f} s\n\n"
            f"Speed: {speed:6.2f} m/s\n"
            f"Target: {target_speed:6.2f} m/s\n\n"
            f"Path Error:\n"
            f"{error:6.3f} m\n\n"
            f"Steering:\n"
            f"{steering:6.2f} deg\n\n"
            f"Lateral Acc:\n"
            f"{lateral_acc:6.3f} m/s^2\n\n"
            f"Final RMSE:\n"
            f"{metrics['path_rmse']:.3f} m\n\n"
            f"Completed:\n"
            f"{metrics['completed']}"
        )

        info_text.set_text(
            information
        )

        return (
            trajectory_line,
            vehicle,
            info_text
        )

    # --------------------------------------
    # BUILD ANIMATION
    # --------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / FPS,
        blit=False,
        repeat=False
    )

    output_path = (
        OUTPUT_DIR
        / f"{SCENARIO}_animation.gif"
    )

    print("Rendering animation...")
    print("Total frames:", len(frame_indices))

    animation.save(
        output_path,
        writer=PillowWriter(fps=FPS),
        dpi=90
    )

    print("\nAnimation saved:")
    print(output_path)

    print("\nSimulation metrics:")

    print(
        "RMSE:",
        f"{metrics['path_rmse']:.3f} m"
    )

    print(
        "Max Error:",
        f"{metrics['max_path_error']:.3f} m"
    )

    print(
        "Max Lateral Acc:",
        f"{metrics['max_lateral_acceleration']:.3f} m/s^2"
    )

    print(
        "Completed:",
        metrics["completed"]
    )

    plt.show()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    create_animation()
