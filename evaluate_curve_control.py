
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from integrated_curve_control import (
    run_simulation,
    MAX_LATERAL_ACCELERATION,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

START_DISTANCE = 20.0
END_DISTANCE = 200.0
BIN_SIZE = 2.0

RESULTS_DIR = (
    Path(__file__).resolve().parent / "results"
)

RESULTS_DIR.mkdir(
    parents=True, exist_ok=True
)


# ==========================================
# 2. VALIDATE SIMULATION DATA
# ==========================================

def validate_history(history, name):

    required = [
        "time",
        "x",
        "y",
        "speed",
        "target_speed",
        "path_error",
        "lateral_acceleration",
        "distance"
    ]

    for key in required:

        if key not in history:
            raise ValueError(
                f"{name}: Missing {key}"
            )

        if not np.all(
            np.isfinite(history[key])
        ):
            raise ValueError(
                f"{name}: Invalid values in {key}"
            )

    n = len(history["distance"])

    if n == 0:
        raise ValueError(
            f"{name}: Empty simulation"
        )

    for key in [
        "target_speed",
        "path_error",
        "lateral_acceleration"
    ]:
        if len(history[key]) != n:
            raise ValueError(
                f"{name}: Length mismatch in {key}"
            )

    if len(history["speed"]) != n + 1:
        raise ValueError(
            f"{name}: Speed length mismatch"
        )

    if np.any(history["speed"] < -1e-6):
        raise ValueError(
            f"{name}: Negative speed detected"
        )

    if np.any(
        np.diff(history["distance"]) < -1e-6
    ):
        raise ValueError(
            f"{name}: Path progress moved backward"
        )

    print(f"{name}: Validation passed")


# ==========================================
# 3. SPATIAL BIN EVALUATION
# ==========================================

def evaluate_by_distance(history, name):

    s = history["distance"]

    path_error = history["path_error"]

    lateral_acceleration = np.abs(
        history["lateral_acceleration"]
    )

    actual_speed = history["speed"][:-1]
    target_speed = history["target_speed"]

    bin_edges = np.arange(
        START_DISTANCE,
        END_DISTANCE + BIN_SIZE,
        BIN_SIZE
    )

    bin_centers = (
        bin_edges[:-1] + bin_edges[1:]
    ) / 2.0

    count = len(bin_centers)

    error_rms = np.full(count, np.nan)
    max_error = np.full(count, np.nan)
    lateral_max = np.full(count, np.nan)
    speed_error_rms = np.full(count, np.nan)

    for i in range(count):

        mask = (
            (s >= bin_edges[i])
            & (s < bin_edges[i + 1])
        )

        if not np.any(mask):
            continue

        error_rms[i] = np.sqrt(
            np.mean(path_error[mask] ** 2)
        )

        max_error[i] = np.max(
            path_error[mask]
        )

        lateral_max[i] = np.max(
            lateral_acceleration[mask]
        )

        speed_error_rms[i] = np.sqrt(
            np.mean(
                (
                    target_speed[mask]
                    - actual_speed[mask]
                ) ** 2
            )
        )

    valid = np.isfinite(error_rms)

    coverage = float(np.mean(valid))

    return {
        "name": name,
        "bin_centers": bin_centers,
        "error_rms": error_rms,
        "max_error": max_error,
        "lateral_max": lateral_max,
        "speed_error_rms": speed_error_rms,
        "valid": valid,
        "coverage": coverage
    }


# ==========================================
# 4. CALCULATE METRICS
# ==========================================

def calculate_metrics(result, mask):

    if not np.any(mask):
        raise RuntimeError(
            "No shared road segments available"
        )

    return {
        "path_rmse": float(
            np.sqrt(
                np.mean(
                    result["error_rms"][mask] ** 2
                )
            )
        ),
        "max_path_error": float(
            np.max(
                result["max_error"][mask]
            )
        ),
        "max_lateral_acceleration": float(
            np.max(
                result["lateral_max"][mask]
            )
        ),
        "speed_rmse": float(
            np.sqrt(
                np.mean(
                    result["speed_error_rms"][mask] ** 2
                )
            )
        )
    }


# ==========================================
# 5. SAVE CSV REPORT
# ==========================================

def save_report(rows):

    output_path = (
        RESULTS_DIR / "curve_evaluation.csv"
    )

    fieldnames = [
        "name",
        "coverage",
        "shared_coverage",
        "path_rmse",
        "max_path_error",
        "max_lateral_acceleration",
        "speed_rmse"
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nCSV saved:", output_path)


# ==========================================
# 6. MAIN EXPERIMENT
# ==========================================

def main():

    print("\n" + "=" * 55)
    print("FAIR CURVE CONTROL EVALUATION")
    print("=" * 55)

    fixed_history = run_simulation(
        use_speed_planning=False
    )

    planned_history = run_simulation(
        use_speed_planning=True
    )

    validate_history(
        fixed_history, "Fixed Speed"
    )

    validate_history(
        planned_history, "Speed Planning"
    )

    fixed = evaluate_by_distance(
        fixed_history, "Fixed Speed"
    )

    planned = evaluate_by_distance(
        planned_history, "Speed Planning"
    )

    # Compare only bins visited by both cars.
    shared_mask = (
        fixed["valid"] & planned["valid"]
    )

    shared_coverage = float(
        np.mean(shared_mask)
    )

    print("\n=== ROAD COVERAGE ===")

    print(
        f"Fixed Speed: "
        f"{fixed['coverage'] * 100:.1f}%"
    )

    print(
        f"Speed Planning: "
        f"{planned['coverage'] * 100:.1f}%"
    )

    print(
        f"Shared Coverage: "
        f"{shared_coverage * 100:.1f}%"
    )

    if shared_coverage < 0.95:
        print(
            "WARNING: Shared coverage is below 95%."
        )
        print(
            "Results describe only the shared bins."
        )

    fixed_metrics = calculate_metrics(
        fixed, shared_mask
    )

    planned_metrics = calculate_metrics(
        planned, shared_mask
    )

    print("\n=== FAIR COMPARISON ===")

    print(
        f"{'Metric':<30}"
        f"{'Fixed':>14}"
        f"{'Planned':>14}"
    )

    for key in fixed_metrics:
        print(
            f"{key:<30}"
            f"{fixed_metrics[key]:>14.3f}"
            f"{planned_metrics[key]:>14.3f}"
        )

    rows = []

    for result, metrics in [
        (fixed, fixed_metrics),
        (planned, planned_metrics)
    ]:

        rows.append({
            "name": result["name"],
            "coverage": result["coverage"],
            "shared_coverage": shared_coverage,
            **metrics
        })

    save_report(rows)

    # ======================================
    # FIGURE 1: PATH ERROR
    # ======================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        fixed["bin_centers"],
        np.where(
            shared_mask,
            fixed["error_rms"],
            np.nan
        ),
        label="Fixed Speed"
    )

    plt.plot(
        planned["bin_centers"],
        np.where(
            shared_mask,
            planned["error_rms"],
            np.nan
        ),
        label="Speed Planning"
    )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel("Path Error RMS (m)")
    plt.title(
        "Path Error on Shared Road Segments"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fair_path_error.png",
        dpi=200
    )

    # ======================================
    # FIGURE 2: LATERAL ACCELERATION
    # ======================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        fixed["bin_centers"],
        np.where(
            shared_mask,
            fixed["lateral_max"],
            np.nan
        ),
        label="Fixed Speed"
    )

    plt.plot(
        planned["bin_centers"],
        np.where(
            shared_mask,
            planned["lateral_max"],
            np.nan
        ),
        label="Speed Planning"
    )

    plt.axhline(
        MAX_LATERAL_ACCELERATION,
        color="red",
        linestyle="--",
        label="Planning Limit"
    )

    plt.xlabel("Distance Along Path (m)")
    plt.ylabel(
        "Maximum Absolute Lateral Acceleration (m/s^2)"
    )
    plt.title(
        "Lateral Acceleration on Shared Road Segments"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "fair_lateral_acceleration.png",
        dpi=200
    )

    print("\nEvaluation completed.")
    print("Results directory:", RESULTS_DIR)

    plt.show()


if __name__ == "__main__":
    main()
