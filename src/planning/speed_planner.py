import numpy as np


class SpeedPlanner:
    def __init__(
        self,
        max_speed=10.0,
        max_lateral_acceleration=0.5,
        max_braking=3.0,
        max_acceleration=2.0,
        min_speed=0.0
    ):
        if max_speed <= 0:
            raise ValueError("max_speed must be positive")

        if max_lateral_acceleration <= 0:
            raise ValueError(
                "max_lateral_acceleration must be positive"
            )

        if max_braking <= 0:
            raise ValueError("max_braking must be positive")

        if max_acceleration <= 0:
            raise ValueError(
                "max_acceleration must be positive"
            )

        if not 0 <= min_speed <= max_speed:
            raise ValueError("Invalid min_speed")

        self.max_speed = max_speed
        self.max_lateral_acceleration = (
            max_lateral_acceleration
        )
        self.max_braking = max_braking
        self.max_acceleration = max_acceleration
        self.min_speed = min_speed

    def calculate_curvature(self, path_x, path_y):
        x = np.asarray(path_x, dtype=float)
        y = np.asarray(path_y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Path must be 1D")

        if len(x) != len(y) or len(x) < 3:
            raise ValueError(
                "Path must have at least 3 matching points"
            )

        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Path contains non-finite values")

        ds = np.hypot(np.diff(x), np.diff(y))

        if np.any(ds <= 0):
            raise ValueError(
                "Consecutive path points must differ"
            )

        s = np.concatenate(([0.0], np.cumsum(ds)))

        dx = np.gradient(x, s)
        dy = np.gradient(y, s)

        ddx = np.gradient(dx, s)
        ddy = np.gradient(dy, s)

        denominator = (dx**2 + dy**2) ** 1.5

        curvature = np.abs(
            dx * ddy - dy * ddx
        ) / np.maximum(denominator, 1e-9)

        return curvature, s

    def plan(self, path_x, path_y, initial_speed=None):

        curvature, s = self.calculate_curvature(
            path_x, path_y
        )

        # 1. Speed limit from curvature
        safe_curvature = np.maximum(curvature, 1e-9)

        curvature_speed = np.sqrt(
            self.max_lateral_acceleration / safe_curvature
        )

        curvature_speed = np.minimum(
            curvature_speed,
            self.max_speed
        )

        speed_profile = curvature_speed.copy()

        # 2. Backward pass: braking constraint
        for i in range(len(speed_profile) - 2, -1, -1):
            ds = s[i + 1] - s[i]

            allowed_speed = np.sqrt(
                speed_profile[i + 1] ** 2
                + 2 * self.max_braking * ds
            )

            speed_profile[i] = min(
                speed_profile[i],
                allowed_speed
            )

        # 3. Optional initial-speed boundary
        if initial_speed is not None:
            if not np.isfinite(initial_speed) or initial_speed < 0:
                raise ValueError("Invalid initial_speed")

            if initial_speed > speed_profile[0] + 1e-9:
                raise ValueError(
                    "Initial speed exceeds planned starting limit"
                )

            speed_profile[0] = initial_speed

        # 4. Forward pass: acceleration constraint
        for i in range(1, len(speed_profile)):
            ds = s[i] - s[i - 1]

            allowed_speed = np.sqrt(
                speed_profile[i - 1] ** 2
                + 2 * self.max_acceleration * ds
            )

            speed_profile[i] = min(
                speed_profile[i],
                allowed_speed
            )

        return {
            "curvature": curvature,
            "distance": s,
            "curvature_speed": curvature_speed,
            "speed_profile": speed_profile
        }
