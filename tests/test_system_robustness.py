
import unittest
import numpy as np

from src.vehicle_model.bicycle_model import BicycleModel
from src.controllers.pure_pursuit import PurePursuitController
from src.controllers.steering_rate_limiter import SteeringRateLimiter
from src.controllers.pid_speed import PIDSpeedController
from src.planning.speed_planner import SpeedPlanner


# ==========================================
# 1. BICYCLE MODEL TESTS
# ==========================================

class TestBicycleModel(unittest.TestCase):

    def test_straight_motion(self):

        vehicle = BicycleModel(wheelbase=2.7)

        vehicle.x = 0.0
        vehicle.y = 0.0
        vehicle.yaw = 0.0
        vehicle.velocity = 5.0

        vehicle.update(
            acceleration=0.0,
            steering_angle=0.0,
            dt=0.1
        )

        self.assertAlmostEqual(
            vehicle.x, 0.5, places=5
        )
        self.assertAlmostEqual(
            vehicle.y, 0.0, places=5
        )
        self.assertAlmostEqual(
            vehicle.yaw, 0.0, places=5
        )

    def test_zero_speed(self):

        vehicle = BicycleModel(wheelbase=2.7)

        vehicle.x = 0.0
        vehicle.y = 0.0
        vehicle.yaw = 0.0
        vehicle.velocity = 0.0

        vehicle.update(
            acceleration=0.0,
            steering_angle=0.2,
            dt=0.1
        )

        self.assertAlmostEqual(vehicle.x, 0.0)
        self.assertAlmostEqual(vehicle.y, 0.0)
        self.assertAlmostEqual(vehicle.yaw, 0.0)

    def test_turning(self):

        vehicle = BicycleModel(wheelbase=2.7)

        vehicle.velocity = 5.0
        vehicle.yaw = 0.0

        vehicle.update(
            acceleration=0.0,
            steering_angle=0.2,
            dt=0.1
        )

        self.assertGreater(vehicle.yaw, 0.0)


# ==========================================
# 2. PURE PURSUIT TESTS
# ==========================================

class TestPurePursuit(unittest.TestCase):

    def setUp(self):

        self.controller = PurePursuitController(
            wheelbase=2.7
        )

    def test_straight_target(self):

        steering = self.controller.calculate_steering(
            x=0.0,
            y=0.0,
            yaw=0.0,
            target_x=10.0,
            target_y=0.0
        )

        self.assertAlmostEqual(
            steering, 0.0, places=5
        )

    def test_left_target(self):

        steering = self.controller.calculate_steering(
            x=0.0,
            y=0.0,
            yaw=0.0,
            target_x=10.0,
            target_y=2.0
        )

        self.assertGreater(steering, 0.0)

    def test_right_target(self):

        steering = self.controller.calculate_steering(
            x=0.0,
            y=0.0,
            yaw=0.0,
            target_x=10.0,
            target_y=-2.0
        )

        self.assertLess(steering, 0.0)

    def test_lookahead_increases_with_speed(self):

        low = self.controller.calculate_lookahead(
            2.0
        )

        high = self.controller.calculate_lookahead(
            10.0
        )

        self.assertGreater(high, low)


# ==========================================
# 3. STEERING RATE LIMITER TESTS
# ==========================================

class TestSteeringRateLimiter(unittest.TestCase):

    def test_rate_limit(self):

        limiter = SteeringRateLimiter(
            max_rate_deg=60.0
        )

        result = limiter.update(
            desired_steering=np.radians(30.0),
            dt=0.1
        )

        # Maximum change = 60 deg/s * 0.1 s
        # = 6 degrees
        self.assertLessEqual(
            abs(result),
            np.radians(6.0) + 1e-9
        )

    def test_gradual_response(self):

        limiter = SteeringRateLimiter(
            max_rate_deg=60.0
        )

        first = limiter.update(
            np.radians(30.0),
            0.1
        )

        second = limiter.update(
            np.radians(30.0),
            0.1
        )

        self.assertGreater(second, first)


# ==========================================
# 4. PID SPEED CONTROLLER TESTS
# ==========================================

class TestPIDSpeedController(unittest.TestCase):

    def setUp(self):

        self.controller = PIDSpeedController(
            kp=1.0,
            ki=0.1,
            kd=0.05,
            min_acceleration=-3.0,
            max_acceleration=2.0,
            anti_windup=True
        )

    def test_acceleration(self):

        output = self.controller.update(
            target_speed=10.0,
            current_speed=0.0,
            dt=0.1
        )

        self.assertGreater(output, 0.0)

    def test_braking(self):

        output = self.controller.update(
            target_speed=0.0,
            current_speed=10.0,
            dt=0.1
        )

        self.assertLess(output, 0.0)

    def test_output_limits(self):

        output = self.controller.update(
            target_speed=100.0,
            current_speed=0.0,
            dt=0.1
        )

        self.assertLessEqual(output, 2.0)
        self.assertGreaterEqual(output, -3.0)


# ==========================================
# 5. SPEED PLANNER TESTS
# ==========================================

class TestSpeedPlanner(unittest.TestCase):

    def setUp(self):

        self.planner = SpeedPlanner(
            max_speed=10.0,
            max_lateral_acceleration=0.5,
            max_braking=3.0,
            max_acceleration=2.0
        )

    def test_straight_path(self):

        x = np.linspace(0.0, 100.0, 501)
        y = np.zeros_like(x)

        result = self.planner.plan(x, y)

        speed = np.asarray(
            result["speed_profile"]
        )

        self.assertEqual(len(speed), len(x))
        self.assertTrue(np.all(np.isfinite(speed)))
        self.assertTrue(np.all(speed >= 0.0))
        self.assertTrue(np.all(speed <= 10.0 + 1e-6))

    def test_curved_path(self):

        x = np.linspace(0.0, 100.0, 501)
        y = 5.0 * np.sin(0.08 * x)

        result = self.planner.plan(x, y)

        speed = np.asarray(
            result["speed_profile"]
        )

        self.assertTrue(np.all(np.isfinite(speed)))
        self.assertTrue(np.all(speed >= 0.0))
        self.assertTrue(np.all(speed <= 10.0 + 1e-6))

        self.assertLess(
            np.min(speed),
            np.max(speed)
        )

    def test_distance_monotonic(self):

        x = np.linspace(0.0, 100.0, 501)
        y = 2.0 * np.sin(0.05 * x)

        result = self.planner.plan(x, y)

        distance = np.asarray(
            result["distance"]
        )

        self.assertTrue(
            np.all(np.diff(distance) >= -1e-9)
        )


# ==========================================
# 6. INTEGRATION TEST
# ==========================================

class TestIntegration(unittest.TestCase):

    def test_complete_control_loop(self):

        vehicle = BicycleModel(wheelbase=2.7)

        vehicle.x = 0.0
        vehicle.y = 0.0
        vehicle.yaw = 0.0
        vehicle.velocity = 0.0

        lateral = PurePursuitController(
            wheelbase=2.7
        )

        limiter = SteeringRateLimiter(
            max_rate_deg=60.0
        )

        longitudinal = PIDSpeedController(
            kp=1.0,
            ki=0.1,
            kd=0.05,
            min_acceleration=-3.0,
            max_acceleration=2.0,
            anti_windup=True
        )

        dt = 0.1

        for _ in range(100):

            steering_desired = (
                lateral.calculate_steering(
                    vehicle.x,
                    vehicle.y,
                    vehicle.yaw,
                    vehicle.x + 10.0,
                    0.0
                )
            )

            steering = limiter.update(
                steering_desired,
                dt
            )

            acceleration = longitudinal.update(
                target_speed=5.0,
                current_speed=vehicle.velocity,
                dt=dt
            )

            vehicle.update(
                acceleration=acceleration,
                steering_angle=steering,
                dt=dt
            )

            state = np.array([
                vehicle.x,
                vehicle.y,
                vehicle.yaw,
                vehicle.velocity
            ])

            self.assertTrue(
                np.all(np.isfinite(state))
            )

        self.assertGreater(vehicle.x, 0.0)
        self.assertGreater(vehicle.velocity, 0.0)
        self.assertLess(
            abs(vehicle.y),
            1.0
        )


if __name__ == "__main__":
    unittest.main()
