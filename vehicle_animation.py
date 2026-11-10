
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from stress_testing import (
    SCENARIOS,
    run_scenario
)


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "results" / "animation"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SCENARIO_NAME = "Sharp Curve"

# Number of simulation steps between
# consecutive animation frames.
FRAME_SKIP = 3

# Playback speed in frames per second.
FPS = 20

CAR_LENGTH = 4.2
CAR_WIDTH = 1.8


# ==========================================
# VEHICLE SHAPE
# ==========================================

def vehicle_corners(x, y, yaw):

    local_corners = np.array([
        [CAR_LENGTH / 2, CAR_WIDTH / 2],
        [CAR_LENGTH / 2, -CAR_WIDTH / 2],
        [-CAR_LENGTH / 2, -CAR_WIDTH / 2],
        [-CAR_LENGTH / 2, CAR_WIDTH / 2]
    ])

    rotation = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    return local_corners @ rotation.T + np.array([x, y])


# ==========================================
# RUN SELECTED SCENARIO
# ==========================================

def get_simulation():

    selected = None

    for config in SCENARIOS:
        if config["name"] == SCENARIO_NAME:
            selected = config
            break

    if selected is None:
        raise ValueError(
            f"Unknown scenario: {SCENARIO_NAME}"
        )

    metrics, history, path_x, path_y = (
        run_scenario(selected)
    )

    return metrics, history, path_x, path_y


# ==========================================
# CREATE ANIMATION
# ==========================================

def create_animation():

    metrics, history, path_x, path_y = (
        get_simulation()
    )

    if len(history["x"]) < 2:
        raise RuntimeError(
            "Not enough simulation data."
        )

    frame_indices = np.arange(
        0,
        len(history["x"]),
        FRAME_SKIP
    )

    if frame_indices[-1] != len(history["x"]) - 1:
        frame_indices = np.append(
            frame_indices,
            len(history["x"]) - 1
        )

    # Approximate vehicle heading from
    # successive trajectory positions.
    x = history["x"]
    y = history["y"]

    dx = np.gradient(x)
    dy = np.gradient(y)

    yaw = np.unwrap(
        np.arctan2(dy, dx)
    )

    fig, (ax_road, ax_info) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={
            "width_ratios": [2.5, 1.0]
        }
    )

    fig.suptitle(
        f"Autonomous Vehicle Simulation - {SCENARIO_NAME}",
        fontsize=14
    )

    # --------------------------------------
    # LEFT: ROAD AND VEHICLE
    # --------------------------------------

    ax_road.plot(
        path_x,
        path_y,
        "k--",
        linewidth=1.5,
        label="Reference Path"
    )

    trajectory_line, = ax_road.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2,
        label="Vehicle Trajectory"
    )

    vehicle_patch = Polygon(
        vehicle_corners(
            x[0],
            y[0],
            yaw[0]
        ),
        closed=True,
        facecolor="tab:red",
        edgecolor="black",
        alpha=0.85
    )

    ax_road.add_patch(vehicle_patch)

    ax_road.set_xlabel("X Position (m)")
    ax_road.set_ylabel("Y Position (m)")

    ax_road.set_title("Path Tracking")
    ax_road.grid(True)
    ax_road.legend(loc="upper right")

    ax_road.set_aspect(
        "equal",
        adjustable="box"
    )

    # --------------------------------------
    # RIGHT: LIVE INFORMATION
    # --------------------------------------

    ax_info.axis("off")

    info_text = ax_info.text(
        0.05,
        0.90,
        "",
        transform=ax_info.transAxes,
        fontsize=12,
        verticalalignment="top",
        family="monospace",
        linespacing=1.8
    )

    # --------------------------------------
    # UPDATE FUNCTION
    # --------------------------------------

    def update(frame_number):

        index = int(
            frame_indices[frame_number]
        )

        trajectory_line.set_data(
            x[:index + 1],
            y[:index + 1]
        )

        corners = vehicle_corners(
            x[index],
            y[index],
            yaw[index]
        )

        vehicle_patch.set_xy(corners)

        # Follow the vehicle with the camera.
        ax_road.set_xlim(
            x[index] - 20,
            x[index] + 40
        )

        ax_road.set_ylim(
            y[index] - 15,
            y[index] + 15
        )

        time_value = history["time"][index]
        speed = history["speed"][index]
        error = history["path_error"][index]
        steering = history["steering_deg"][index]
        lateral = (
            history["lateral_acceleration"][index]
        )

        information = (
            f"Time: {time_value:6.1f} s\n"
            f"\n"
            f"Speed: {speed:6.2f} m/s\n"
            f"Path Error: {error:6.3f} m\n"
            f"Steering: {steering:6.2f} deg\n"
            f"Lateral Acc: {lateral:6.3f} m/s^2\n"
            f"\n"
            f"Scenario: {SCENARIO_NAME}\n"
            f"\n"
            f"Final Path RMSE:\n"
            f"{metrics['path_rmse']:.3f} m\n"
            f"\n"
            f"Completion:\n"
            f"{metrics['completion_percent']:.1f}%"
        )

        info_text.set_text(
            information
        )

        return (
            trajectory_line,
            vehicle_patch,
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
        / "vehicle_simulation.gif"
    )

    print("Rendering animation...")
    print("Frames:", len(frame_indices))

    animation.save(
        output_path,
        writer=PillowWriter(fps=FPS)
    )

    print("Animation saved:", output_path)

    plt.show()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    create_animation()
