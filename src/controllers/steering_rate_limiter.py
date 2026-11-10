import math


class SteeringRateLimiter:
    def __init__(self, max_rate_deg=60.0):
        self.max_rate = math.radians(max_rate_deg)
        self.current_steering = 0.0

    def update(self, desired_steering, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        max_change = self.max_rate * dt

        difference = desired_steering - self.current_steering

        change = max(
            -max_change,
            min(difference, max_change)
        )

        self.current_steering += change

        return self.current_steering
