
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path("results/random_path")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_random_path(
    length=250,
    num_points=1200,
    seed=42,
    amplitude=5.0,
    num_control_points=18
):
    """
    Generate a smooth, irregular road.

    The seed makes the road reproducible.
    """

    rng = np.random.default_rng(seed)

    x = np.linspace(0, length, num_points)

    control_x = np.linspace(
        0, length, num_control_points
    )

    control_y = rng.uniform(
        -amplitude,
        amplitude,
        num_control_points
    )

    # Start near the origin.
    control_y[0] = 0.0

    # Interpolate the random control points.
    y = np.interp(
        x,
        control_x,
        control_y
    )

    # Gaussian smoothing using NumPy only.
    sigma = 20
    radius = 4 * sigma

    offsets = np.arange(
        -radius,
        radius + 1
    )

    kernel = np.exp(
        -0.5 * (offsets / sigma) ** 2
    )

    kernel /= kernel.sum()

    # Edge padding avoids artificial
    # distortion at the road boundaries.
    padded = np.pad(
        y,
        (radius, radius),
        mode="edge"
    )

    y = np.convolve(
        padded,
        kernel,
        mode="valid"
    )

    return x, y


def main():

    configurations = [
        ("Random Smooth", 42, 3.0, 12),
        ("Random Sharp", 52, 6.0, 20),
        ("Random Extreme", 72, 8.0, 25)
    ]

    plt.figure(figsize=(13, 8))

    for index, config in enumerate(
        configurations, start=1
    ):

        name, seed, amplitude, points = config

        x, y = generate_random_path(
            seed=seed,
            amplitude=amplitude,
            num_control_points=points
        )

        plt.subplot(3, 1, index)

        plt.plot(
            x,
            y,
            linewidth=2,
            label=name
        )

        plt.title(name)
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.grid(True)
        plt.legend()

        file_name = name.lower().replace(
            " ", "_"
        )

        np.savez(
            OUTPUT_DIR / f"{file_name}.npz",
            path_x=x,
            path_y=y
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "random_paths.png"
    )

    plt.savefig(
        output_path,
        dpi=180
    )

    print("Random paths generated.")
    print("Saved:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
