class PIDSpeedController:
    def __init__(
        self,
        kp=1.0,
        ki=0.1,
        kd=0.05,
        min_acceleration=-3.0,
        max_acceleration=2.0,
        anti_windup=True
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.min_acceleration = min_acceleration
        self.max_acceleration = max_acceleration

        self.anti_windup = anti_windup

        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, target_speed, current_speed, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")

        # Calculate speed error
        error = target_speed - current_speed

        # Proportional term
        p = self.kp * error

        # Derivative term
        derivative = (error - self.previous_error) / dt
        d = self.kd * derivative

        # Candidate integral
        candidate_integral = self.integral + error * dt

        # Candidate PID output
        candidate_output = (
            p
            + self.ki * candidate_integral
            + d
        )

        # Conditional integration anti-windup
        if self.anti_windup:

            upper_saturated = (
                candidate_output > self.max_acceleration
                and error > 0
            )

            lower_saturated = (
                candidate_output < self.min_acceleration
                and error < 0
            )

            # Only integrate if it does not worsen saturation
            if not (upper_saturated or lower_saturated):
                self.integral = candidate_integral

        else:
            self.integral = candidate_integral

        # Final PID output
        acceleration = (
            p
            + self.ki * self.integral
            + d
        )

        # Acceleration limits
        acceleration = max(
            self.min_acceleration,
            min(acceleration, self.max_acceleration)
        )

        self.previous_error = error

        return acceleration
