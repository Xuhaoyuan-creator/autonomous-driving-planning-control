
from pathlib import Path
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.pid_speed import PIDSpeedController
from src.controllers.steering_rate_limiter import SteeringRateLimiter
from src.planning.speed_planner import SpeedPlanner


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent

PATH_DIR = ROOT / "results" / "random_path"
OUTPUT_DIR = ROOT / "results" / "random_tracking"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "random_smooth",
    "random_sharp",
    "random_extreme",
]

DT = 0.1
MAX_STEPS = 3000

WHEELBASE = 2.7
MAX_SPEED = 8.0

MAX_ACCELERATION = 2.0
MAX_BRAKING = 3.0

TARGET_LATERAL_ACCELERATION = 0.5
LOOKAHEAD_GAIN = 0.2

END_TOLERANCE = 2.0


# ==================================================
# SPEED PLANNER OUTPUT COMPATIBILITY
# ==================================================

def extract_speed_profile(plan_result, path_length):
    """
    Accept:
      1. dict containing a speed profile
      2. 1D NumPy array / list of speeds
      3. tuple/list containing several arrays

    Reject unknown formats rather than silently
    interpreting curvature as speed.
    """

    candidates = []

    if isinstance(plan_result, dict):

        possible_keys = [
            "speed_profile",
            "target_speed",
            "target_speeds",
            "speeds",
            "speed",
            "velocity_profile",
            "v_profile",
        ]

        for key in possible_keys:
            if key in plan_result:
                candidates.append(
                    (key, plan_result[key])
                )

        if not candidates:
            raise ValueError(
                "SpeedPlanner returned a dictionary, "
                "but no recognized speed-profile key "
                "was found. Available keys: "
                f"{list(plan_result.keys())}"
            )

    elif isinstance(plan_result, np.ndarray):

        candidates.append(
            ("array", plan_result)
        )

    elif isinstance(plan_result, (tuple, list)):

        # A simple numeric list can itself be
        # the speed profile.
        try:
            direct = np.asarray(
                plan_result,
                dtype=float
            )

            if (
                direct.ndim == 1
                and direct.size == path_length
            ):
                candidates.append(
                    ("direct", direct)
                )

        except (TypeError, ValueError):
            pass

        # Existing project convention:
        # (speed_profile, distance,
        #  curvature, curvature_speed)
        if not candidates and len(plan_result) == 4:
            candidates.append(
                ("first_tuple_element", plan_result[0])
            )

    else:
        raise TypeError(
            "Unsupported SpeedPlanner output type: "
            f"{type(plan_result)}"
        )

    for source, candidate in candidates:

        try:
            profile = np.asarray(
                candidate,
                dtype=float
            ).reshape(-1)

        except (TypeError, ValueError):
            continue

        if profile.size != path_length:
            continue

        if not np.all(np.isfinite(profile)):
            continue

        if np.any(profile < 0):
            continue

        print(
            "Speed profile loaded from:",
            source
        )

        print(
            "Speed range:",
            f"{profile.min():.3f}",
            "to",
            f"{profile.max():.3f}",
            "m/s"
        )

        return profile

    raise ValueError(
        "Could not extract a valid speed profile. "
        f"Expected {path_length} finite, "
        "non-negative numeric values. "
        "Check SpeedPlanner.plan() output."
    )


# ==================================================
# GEOMETRY
# ==================================================

def nearest_path_index(x, y, path_x, path_y):

    distances_squared = (
        (path_x - x) ** 2
        + (path_y - y) ** 2
    )

    return int(
        np.argmin(distances_squared)
    )


def calculate_path_error(x, y, path_x, path_y):

    distances = np.hypot(
        path_x - x,
        path_y - y
    )

    return float(
        np.min(distances)
    )


def calculate_lateral_acceleration(
    velocity,
    steering_angle
):

    return abs(
        velocity ** 2
        * math.tan(steering_angle)
        / WHEELBASE
    )


def find_target_index(
    x,
    y,
    nearest_index,
    lookahead,
    path_x,
    path_y
):

    target_index = nearest_index

    while target_index < len(path_x) - 1:

        distance = math.hypot(
            path_x[target_index] - x,
            path_y[target_index] - y
        )

        if distance >= lookahead:
            break

        target_index += 1

    return target_index


# ==================================================
# SIMULATION
# ==================================================

def run_simulation(name):

    path_file = PATH_DIR / f"{name}.npz"

    if not path_file.exists():
        raise FileNotFoundError(
            f"Missing path: {path_file}\n"
            "Run random_path_generator.py first."
        )

    with np.load(path_file) as data:

        path_x = np.asarray(
            data["path_x"],
            dtype=float
        )

        path_y = np.asarray(
            data["path_y"],
            dtype=float
        )

    if len(path_x) < 3:
        raise ValueError("Path is too short.")

    initial_yaw = math.atan2(
        path_y[1] - path_y[0],
        path_x[1] - path_x[0]
    )

    # ------------------------------------------
    # Vehicle
    # ------------------------------------------

    vehicle = BicycleModel(
        wheelbase=WHEELBASE
    )

    vehicle.x = float(path_x[0])
    vehicle.y = float(path_y[0])
    vehicle.yaw = initial_yaw
    vehicle.velocity = 0.0

    # ------------------------------------------
    # Controllers
    # ------------------------------------------

    lateral_controller = PurePursuitController(
        wheelbase=WHEELBASE
    )

    speed_controller = PIDSpeedController(
        kp=1.0,
        ki=0.1,
        kd=0.05,
        min_acceleration=-MAX_BRAKING,
        max_acceleration=MAX_ACCELERATION,
        anti_windup=True
    )

    steering_limiter = SteeringRateLimiter(
        max_rate_deg=60
    )

    speed_planner = SpeedPlanner(
        max_speed=MAX_SPEED,
        max_lateral_acceleration=(
            TARGET_LATERAL_ACCELERATION
        ),
        max_braking=MAX_BRAKING,
        max_acceleration=MAX_ACCELERATION
    )

    # ------------------------------------------
    # Speed planning
    # ------------------------------------------

    plan_result = speed_planner.plan(
        path_x,
        path_y
    )

    speed_profile = extract_speed_profile(
        plan_result,
        len(path_x)
    )

    # ------------------------------------------
    # History
    # ------------------------------------------

    history = {
        "time": [],
        "x": [],
        "y": [],
        "yaw": [],
        "speed": [],
        "target_speed": [],
        "path_error": [],
        "steering_deg": [],
        "lateral_acceleration": [],
        "nearest_index": [],
    }

    completed = False
    numerical_failure = False

    # ------------------------------------------
    # Main simulation loop
    # ------------------------------------------

    for step in range(MAX_STEPS):

        nearest_index = nearest_path_index(
            vehicle.x,
            vehicle.y,
            path_x,
            path_y
        )

        lookahead = (
            2.0
            + LOOKAHEAD_GAIN
            * abs(vehicle.velocity)
        )

        target_index = find_target_index(
            vehicle.x,
            vehicle.y,
            nearest_index,
            lookahead,
            path_x,
            path_y
        )

        desired_steering = (
            lateral_controller.calculate_steering(
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                float(path_x[target_index]),
                float(path_y[target_index])
            )
        )

        steering = steering_limiter.update(
            desired_steering,
            DT
        )

        target_speed = float(
            speed_profile[nearest_index]
        )

        acceleration = speed_controller.update(
            target_speed,
            vehicle.velocity,
            DT
        )

        vehicle.update(
            acceleration=acceleration,
            steering_angle=steering,
            dt=DT
        )

        error = calculate_path_error(
            vehicle.x,
            vehicle.y,
            path_x,
            path_y
        )

        lateral_acc = (
            calculate_lateral_acceleration(
                vehicle.velocity,
                steering
            )
        )

        values = [
            vehicle.x,
            vehicle.y,
            vehicle.yaw,
            vehicle.velocity,
            steering,
            error,
            lateral_acc,
        ]

        if not all(
            math.isfinite(value)
            for value in values
        ):
            numerical_failure = True
            break

        history["time"].append(
            (step + 1) * DT
        )

        history["x"].append(
            vehicle.x
        )

        history["y"].append(
            vehicle.y
        )

        history["yaw"].append(
            vehicle.yaw
        )

        history["speed"].append(
            vehicle.velocity
        )

        history["target_speed"].append(
            target_speed
        )

        history["path_error"].append(
            error
        )

        history["steering_deg"].append(
            math.degrees(steering)
        )

        history["lateral_acceleration"].append(
            lateral_acc
        )

        history["nearest_index"].append(
            nearest_index
        )

        distance_to_end = math.hypot(
            vehicle.x - path_x[-1],
            vehicle.y - path_y[-1]
        )

        if (
            nearest_index >= len(path_x) - 3
            and distance_to_end <= END_TOLERANCE
        ):
            completed = True
            break

    # ------------------------------------------
    # Convert history to arrays
    # ------------------------------------------

    for key in history:

        history[key] = np.asarray(
            history[key]
        )

    if len(history["x"]) == 0:
        raise RuntimeError(
            f"No valid simulation states: {name}"
        )

    # ------------------------------------------
    # Performance metrics
    # ------------------------------------------

    errors = history["path_error"]

    rmse = math.sqrt(
        float(
            np.mean(errors ** 2)
        )
    )

    max_error = float(
        np.max(errors)
    )

    max_lateral_acc = float(
        np.max(
            history["lateral_acceleration"]
        )
    )

    progress = (
        100.0
        * max(history["nearest_index"])
        / (len(path_x) - 1)
    )

    metrics = {
        "scenario": name,
        "path_rmse": rmse,
        "max_path_error": max_error,
        "max_lateral_acceleration": max_lateral_acc,
        "max_progress_percent": progress,
        "completed": completed,
        "numerical_failure": numerical_failure,
        "duration": float(
            history["time"][-1]
        ),
    }

    return (
        metrics,
        history,
        path_x,
        path_y,
        speed_profile
    )


# ==================================================
# PLOTS
# ==================================================

def plot_trajectory(
    name,
    history,
    path_x,
    path_y
):

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        path_x,
        path_y,
        "k--",
        linewidth=1.5,
        label="Reference Path"
    )

    ax.plot(
        history["x"],
        history["y"],
        color="tab:blue",
        linewidth=2,
        label="Actual Trajectory"
    )

    ax.scatter(
        history["x"][0],
        history["y"][0],
        color="green",
        s=55,
        label="Start"
    )

    ax.scatter(
        path_x[-1],
        path_y[-1],
        color="red",
        s=55,
        label="Destination"
    )

    ax.set_title(
        f"Random Path Tracking - {name}"
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    ax.grid(True)
    ax.legend()
    ax.axis("equal")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{name}_trajectory.png",
        dpi=180
    )

    plt.close(fig)


def plot_performance(name, history):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True
    )

    axes[0].plot(
        history["time"],
        history["path_error"]
    )

    axes[0].set_ylabel(
        "Path Error (m)"
    )

    axes[0].grid(True)

    axes[1].plot(
        history["time"],
        history["speed"],
        label="Actual Speed"
    )

    axes[1].plot(
        history["time"],
        history["target_speed"],
        "--",
        label="Target Speed"
    )

    axes[1].set_ylabel(
        "Speed (m/s)"
    )

    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        history["time"],
        history["lateral_acceleration"],
        label="Actual Lateral Acceleration"
    )

    axes[2].axhline(
        TARGET_LATERAL_ACCELERATION,
        color="red",
        linestyle="--",
        label="Planning Target"
    )

    axes[2].set_ylabel(
        "Lateral Acc. (m/s^2)"
    )

    axes[2].set_xlabel(
        "Time (s)"
    )

    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(
        f"Tracking Performance - {name}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{name}_performance.png",
        dpi=180
    )

    plt.close(fig)


# ==================================================
# MAIN
# ==================================================

def main():

    all_metrics = []

    print("=" * 65)
    print("RANDOM PATH TRACKING CHALLENGE")
    print("=" * 65)

    for name in SCENARIOS:

        print(f"\nRunning: {name}")

        (
            metrics,
            history,
            path_x,
            path_y,
            speed_profile
        ) = run_simulation(name)

        all_metrics.append(metrics)

        plot_trajectory(
            name,
            history,
            path_x,
            path_y
        )

        plot_performance(
            name,
            history
        )

        print(
            "  RMSE:",
            f"{metrics['path_rmse']:.3f} m"
        )

        print(
            "  Max Error:",
            f"{metrics['max_path_error']:.3f} m"
        )

        print(
            "  Max Lateral Acc:",
            f"{metrics['max_lateral_acceleration']:.3f} m/s^2"
        )

        print(
            "  Progress:",
            f"{metrics['max_progress_percent']:.1f}%"
        )

        print(
            "  Completed:",
            metrics["completed"]
        )

        print(
            "  Numerical Failure:",
            metrics["numerical_failure"]
        )

    csv_path = (
        OUTPUT_DIR
        / "random_tracking_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_metrics[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            all_metrics
        )

    print("\nResults saved:")
    print(csv_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
