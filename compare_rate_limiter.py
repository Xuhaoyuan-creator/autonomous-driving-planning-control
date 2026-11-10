
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_adaptive import run_simulation


def main():

    # 1. Run two simulations
    # Original controller: no steering rate limit
    original = run_simulation(
        "adaptive",
        use_rate_limiter=False
    )

    # Controller with steering rate limiter
    limited = run_simulation(
        "adaptive",
        use_rate_limiter=True
    )

    # 2. Create results folder
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)

    # 3. Compare tracking accuracy
    print("\n=== Tracking Accuracy ===")
    print(f"Original RMSE: {original['rmse']:.3f} m")
    print(f"Limited RMSE: {limited['rmse']:.3f} m")

    print("\n=== Maximum Tracking Error ===")
    print(f"Original: {original['max_error']:.3f} m")
    print(f"Limited: {limited['max_error']:.3f} m")

    # 4. Calculate steering rate
    # Initial steering angle is 0 degrees
    original_steering = np.array(
        [0.0] + original["steering"]
    )

    limited_steering = np.array(
        [0.0] + limited["steering"]
    )

    dt = 0.1

    original_rate = np.diff(original_steering) / dt
    limited_rate = np.diff(limited_steering) / dt

    print("\n=== Maximum Steering Rate ===")
    print(
        f"Original: {np.max(np.abs(original_rate)):.2f} deg/s"
    )
    print(
        f"Limited: {np.max(np.abs(limited_rate)):.2f} deg/s"
    )

    # 5. Plot steering angle comparison
    plt.figure(figsize=(11, 6))

    plt.plot(
        original["time"][:-1],
        original["steering"],
        label="Original Adaptive"
    )

    plt.plot(
        limited["time"][:-1],
        limited["steering"],
        label="Adaptive + Rate Limiter"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle (deg)")
    plt.title("Steering Angle Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "rate_limiter_steering.png",
        dpi=200
    )

    # 6. Plot steering rate comparison
    plt.figure(figsize=(11, 6))

    plt.plot(
        original["time"][:-1],
        original_rate,
        label="Original Adaptive"
    )

    plt.plot(
        limited["time"][:-1],
        limited_rate,
        label="Adaptive + Rate Limiter"
    )

    plt.axhline(
        60,
        color="red",
        linestyle="--",
        label="+60 deg/s limit"
    )

    plt.axhline(
        -60,
        color="red",
        linestyle="--",
        label="-60 deg/s limit"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Steering Rate (deg/s)")
    plt.title("Steering Rate Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "rate_limiter_rate.png",
        dpi=200
    )

    # 7. Plot tracking error comparison
    plt.figure(figsize=(11, 6))

    plt.plot(
        original["time"][1:],
        original["error"],
        label="Original Adaptive"
    )

    plt.plot(
        limited["time"][1:],
        limited["error"],
        label="Adaptive + Rate Limiter"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.title("Tracking Error Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "rate_limiter_error.png",
        dpi=200
    )

    # 8. Show figures
    plt.show()


if __name__ == "__main__":
    main()
