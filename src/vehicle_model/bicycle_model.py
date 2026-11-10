import math


class BicycleModel:
    def __init__(self, wheelbase=2.7):
        self.wheelbase = wheelbase

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.velocity = 0.0

    def update(self, acceleration, steering_angle, dt):
        # Calculate velocity components
        x_dot = self.velocity * math.cos(self.yaw)
        y_dot = self.velocity * math.sin(self.yaw)

        # Calculate yaw rate
        yaw_dot = (
            self.velocity / self.wheelbase
        ) * math.tan(steering_angle)

        # Update vehicle state
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.yaw += yaw_dot * dt
        self.velocity += acceleration * dt

    def get_state(self):
        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "velocity": self.velocity
        }