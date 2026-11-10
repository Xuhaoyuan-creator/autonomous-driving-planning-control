import math


class PurePursuitController:

    def __init__(
        self,
        wheelbase=2.7,
        max_steering_deg=30,
        base_lookahead=2.0,
        lookahead_gain=0.6
    ):
        self.wheelbase = wheelbase
        self.max_steering = math.radians(max_steering_deg)

        self.base_lookahead = base_lookahead
        self.lookahead_gain = lookahead_gain

    def calculate_lookahead(self, velocity):
        return self.base_lookahead + self.lookahead_gain * abs(velocity)

    def calculate_steering(self, x, y, yaw, target_x, target_y):
        dx = target_x - x
        dy = target_y - y

        lookahead_distance = math.hypot(dx, dy)

        if lookahead_distance < 1e-6:
            return 0.0

        alpha = math.atan2(dy, dx) - yaw

        steering = math.atan2(
            2 * self.wheelbase * math.sin(alpha),
            lookahead_distance
        )

        steering = max(
            -self.max_steering,
            min(steering, self.max_steering)
        )

        return steering