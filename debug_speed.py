
import numpy as np

from src.planning.speed_planner import SpeedPlanner
from src.controllers.pid_speed import PIDSpeedController
from src.vehicle_model.bicycle_model import BicycleModel


print("\n========== FILE CHECK ==========")

import src.planning.speed_planner as sp
import src.controllers.pid_speed as pid_module
import src.vehicle_model.bicycle_model as vehicle_module

print("Speed Planner:", sp.__file__)
print("PID Controller:", pid_module.__file__)
print("Bicycle Model:", vehicle_module.__file__)


print("\n========== TEST 1: SPEED PLANNER ==========")

path_x = np.linspace(0, 250, 1251)
path_y = 2 * np.sin(0.05 * path_x)

planner = SpeedPlanner(
    max_speed=10.0,
    max_lateral_acceleration=0.5,
    max_braking=3.0,
    max_acceleration=2.0
)

result = planner.plan(path_x, path_y)

speed_profile = np.asarray(
    result["speed_profile"]
)

print("First 10 speeds:", speed_profile[:10])
print("Minimum speed:", np.min(speed_profile))
print("Maximum speed:", np.max(speed_profile))
print("Speed at index 100:", speed_profile[100])

if np.max(speed_profile) <= 0:
    print("FAILED: Speed Planner outputs zero.")
else:
    print("PASSED: Speed Planner outputs positive speeds.")


print("\n========== TEST 2: PID ==========")

pid = PIDSpeedController(
    kp=1.0,
    ki=0.1,
    kd=0.05,
    min_acceleration=-3.0,
    max_acceleration=2.0,
    anti_windup=True
)

acceleration = pid.update(
    target_speed=10.0,
    current_speed=0.0,
    dt=0.1
)

print("Target speed: 10.0")
print("Current speed: 0.0")
print("PID acceleration:", acceleration)

if acceleration > 0:
    print("PASSED: PID produces positive acceleration.")
else:
    print("FAILED: PID does not accelerate.")


print("\n========== TEST 3: VEHICLE MODEL ==========")

vehicle = BicycleModel(wheelbase=2.7)

vehicle.x = 0.0
vehicle.y = 0.0
vehicle.yaw = 0.0
vehicle.velocity = 0.0

print("Initial speed:", vehicle.velocity)

for i in range(10):
    vehicle.update(
        acceleration=2.0,
        steering_angle=0.0,
        dt=0.1
    )

    print(
        f"Step {i + 1}: "
        f"speed={vehicle.velocity:.3f} m/s"
    )

if vehicle.velocity > 0:
    print("PASSED: Vehicle accelerates.")
else:
    print("FAILED: Vehicle model does not accelerate.")


print("\n========== DIAGNOSIS COMPLETE ==========")
