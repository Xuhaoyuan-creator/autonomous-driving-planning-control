import math
from src.vehicle_model.bicycle_model import BicycleModel

def test_bicycle_model():
     vehicle = BicycleModel()
     state = vehicle.get_state()

     assert state["x"] == 0.0
     assert state["y"] == 0.0
     assert state["yaw"] == 0.0
     assert state["velocity"] == 0.0

def test_straight_motion():
     vehicle = BicycleModel()
     vehicle.velocity = 10.0

     vehicle.update(acceleration=0.0, steering_angle=0.0, dt=0.1)
     state = vehicle.get_state()

     assert math.isclose(state["x"],1.0)
     assert math.isclose(state["y"],0.0)
     assert math.isclose(state["yaw"],0.0)


def test_acceleration():
    vehicle = BicycleModel()

    vehicle.update(
        acceleration=2.0,
        steering_angle=0.0,
        dt=0.1
    )

    assert math.isclose(vehicle.velocity, 0.2)

def test_turning():
    vehicle = BicycleModel(wheelbase=2.7)
    vehicle.velocity = 10.0

    vehicle.update(acceleration=0.0, steering_angle=math.radians(5), dt=0.1)
    

    expected_yaw = (10.0 / 2.7* math.tan(math.radians(5))* 0.1)

    assert math.isclose(vehicle.yaw, expected_yaw)
    assert vehicle.yaw > 0