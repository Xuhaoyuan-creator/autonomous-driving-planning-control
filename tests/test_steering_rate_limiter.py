import math
import pytest

from src.controllers.steering_rate_limiter import SteeringRateLimiter


def test_steering_increases_gradually():
    limiter = SteeringRateLimiter(max_rate_deg=60)

    desired = math.radians(30)

    actual = limiter.update(desired, dt=0.1)

    assert math.degrees(actual) == pytest.approx(6.0)


def test_steering_reaches_target():
    limiter = SteeringRateLimiter(max_rate_deg=60)

    desired = math.radians(30)

    for _ in range(5):
        actual = limiter.update(desired, dt=0.1)

    assert math.degrees(actual) == pytest.approx(30.0)


def test_steering_decreases_gradually():
    limiter = SteeringRateLimiter(max_rate_deg=60)

    limiter.current_steering = math.radians(20)

    actual = limiter.update(
        desired_steering=math.radians(-20),
        dt=0.1
    )

    assert math.degrees(actual) == pytest.approx(14.0)


def test_invalid_dt():
    limiter = SteeringRateLimiter()

    with pytest.raises(ValueError):
        limiter.update(0.0, dt=0.0)
