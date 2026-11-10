
# Autonomous Vehicle Planning and Control

A Python-based autonomous vehicle simulation project
integrating path tracking, longitudinal speed control,
and curvature-aware speed planning.

## Vehicle Simulation Demo

![Vehicle Simulation](results/animation/vehicle_simulation.gif)

## Key Features

- Kinematic bicycle vehicle model
- Pure Pursuit lateral controller
- Adaptive lookahead distance
- Steering rate limiter
- PID longitudinal speed controller
- Curvature-based speed planning
- Automated scenario testing
- Controller parameter optimization
- Robustness stress testing

## Experimental Results

### Lookahead Optimization

Five lookahead gains were evaluated.

| Lookahead Gain | Path RMSE |
|---|---|
| 0.20 | 0.064 m |
| 0.40 | 0.065 m |
| 0.60 | 0.073 m |
| 0.80 | 0.093 m |
| 1.00 | 0.126 m |

The best-performing candidate was k = 0.20.

![Lookahead Results](results/lookahead_optimization/lookahead_rmse.png)

### Cross-Scenario Validation

The optimized controller achieved:

- 11.11% average reported RMSE improvement.
- Lower RMSE in all 12 evaluated scenarios.
- Full shared spatial coverage in the evaluated scenarios.

![Validation Results](results/lookahead_validation/validation_rmse.png)

### Robustness Testing

Six stress scenarios were evaluated.

All six passed the preliminary tracking
and completion criteria.

![Stress Test Results](results/stress_testing/stress_trajectories.png)

## Engineering Limitations

The curvature-based speed planner was configured
with a lateral acceleration target of 0.5 m/s^2.

However, the maximum recorded lateral acceleration
reached 6.628 m/s^2 during the Large Offset test.

The current PASS criteria do not enforce a strict
lateral acceleration limit.

The simulation has not been validated on a real vehicle.

## Installation

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

```bash
python scenario_testing.py
python optimize_lookahead.py
python validate_lookahead.py
python stress_testing.py
python vehicle_animation.py
```

## Technical Report

[View Full Performance Report](results/performance_report/PERFORMANCE_REPORT.md)

## Future Improvements

- Dynamic bicycle model
- Lateral acceleration constraints
- Model Predictive Control (MPC)
- ROS 2 integration
- More realistic vehicle dynamics
