Autonomous Vehicle Planning & Control Simulation

A Python-based autonomous vehicle simulation framework for path tracking, longitudinal control, speed planning, controller optimization, and performance evaluation.

This project explores how an autonomous vehicle follows reference trajectories under different road geometries and operating conditions. It integrates a kinematic bicycle model, Pure Pursuit lateral control, PID longitudinal control, steering rate limiting, and curvature-based speed planning.

The project emphasizes not only successful path tracking, but also quantitative evaluation, reproducible experiments, and understanding controller limitations.

1. Project Demo

Random Path Tracking Challenge

The vehicle follows a procedurally generated, smooth but irregular road containing turns of varying direction and curvature.



The animation visualizes the reference path, actual vehicle trajectory, vehicle heading, and live control metrics.

Original Vehicle Simulation



2. Key Features

Kinematic Bicycle Model: Simulates planar vehicle motion using position, heading, velocity, wheelbase, and steering angle.

Pure Pursuit Controller: Computes steering commands based on the vehicle's position, heading, and a lookahead target.

Adaptive Lookahead: Adjusts the lookahead distance according to vehicle speed.

Steering Rate Limiter: Limits how quickly steering commands can change.

PID Speed Controller: Regulates longitudinal vehicle speed.

Curvature-Based Speed Planner: Generates target speeds based on road curvature and acceleration constraints.

Scenario Testing: Evaluates tracking across different path geometries and target speeds.

Parameter Optimization: Compares lookahead settings and validates performance across multiple scenarios.

Stress Testing: Tests performance under challenging initial conditions and road geometries.

Random Path Tracking: Generates reproducible irregular roads and evaluates vehicle tracking performance.

Visualization: Produces trajectory plots, performance charts, CSV results, and animated GIF demonstrations.

3. System Architecture

The simulation follows a modular planning-and-control structure:

Reference Path
      |
      v
Curvature-Based Speed Planner
      |
      v
Target Speed ----------------------+
                                  |
Vehicle State                     |
      |                           |
      +--> Pure Pursuit           +--> PID Speed Control
      |         |                           |
      |    Steering Command           Acceleration
      |         |                           |
      |    Steering Rate Limiter            |
      |         |                           |
      +---------+---------------------------+
                |
                v
        Kinematic Bicycle Model
                |
                v
          Updated Vehicle State
                |
                v
        Performance Evaluation

The vehicle model is simulated in a closed loop, with updated states continuously fed back into the controllers.

4. Vehicle Dynamics Model

A kinematic bicycle model is used to approximate planar vehicle motion.

The governing equations are:

[
\dot{x} = v\cos(\psi)
]

[
\dot{y} = v\sin(\psi)
]

[
\dot{\psi} = \frac{v}{L}\tan(\delta)
]

[
\dot{v} = a
]

Where:

(x,y): Vehicle position

(v): Vehicle speed

(\psi): Vehicle heading

(L): Wheelbase

(\delta): Steering angle

(a): Longitudinal acceleration

The model provides a lightweight environment for evaluating control algorithms without requiring a full vehicle dynamics simulator.

5. Lateral Control: Pure Pursuit

Pure Pursuit calculates a steering command based on a target point ahead of the vehicle.

The adaptive lookahead distance is defined as:

[
L_d = L_{d,0} + kv
]

Where:

(L_{d,0}): Base lookahead distance

(k): Speed-dependent lookahead gain

(v): Vehicle speed

Smaller lookahead distances can improve responsiveness but may introduce aggressive steering behavior. Larger lookahead distances generally produce smoother steering but can increase corner-cutting and tracking error.

A steering rate limiter is also included to restrict abrupt steering changes.

6. Lookahead Parameter Optimization

A parameter sweep was conducted to evaluate different adaptive lookahead gains.

Lookahead Gain (k)

Path Tracking RMSE

0.20

0.064 m

0.40

0.065 m

0.60

0.073 m

0.80

0.093 m

1.00

0.126 m

The best-performing gain in this experiment was k = 0.20.

Cross-Scenario Validation

The optimized lookahead configuration was subsequently evaluated across 12 test conditions.

Improved RMSE in 12 out of 12 tested conditions

Average reported RMSE improvement: 11.11%

Reported spatial path coverage: 100%

These findings apply to the simulated scenarios and should not be interpreted as proof of optimal performance under all driving conditions.

7. Longitudinal Control and Speed Planning

The longitudinal controller uses PID feedback to track the planned vehicle speed.

A curvature-based speed planner reduces the target speed in sharper turns.

The underlying relationship is:

[
a_y = v^2|\kappa|
]

Where:

(a_y): Lateral acceleration

(v): Vehicle speed

(\kappa): Path curvature

The planner uses a lateral acceleration target of 0.5 m/s² in the relevant experiments.

This is a planning target rather than a guaranteed physical limit. Actual simulated lateral acceleration may exceed the target because of tracking deviations, steering dynamics, and controller behavior.

8. Scenario Testing

The simulation evaluates multiple road geometries and vehicle operating conditions.

Road categories include:

Gentle curves

Medium curves

Sharp curves

Test speeds include:

6 m/s

8 m/s

10 m/s

12 m/s

Fixed-speed and planned-speed configurations are compared across 24 test cases.

Performance evaluation includes tracking error, speed behavior, path completion, and lateral acceleration.

9. Random Path Tracking Challenge

To evaluate performance beyond predictable sinusoidal trajectories, the project includes a random-path generation and tracking experiment.

Path Generation

Random control points are generated using reproducible random seeds. The resulting path is interpolated and smoothed to create a continuous reference trajectory.

Three scenarios are available:

Scenario

Description

Random Smooth

Irregular road with relatively gentle turns

Random Sharp

More frequent and pronounced changes in curvature

Random Extreme

More challenging road geometry with stronger directional changes

Trajectory Comparison

The following plots compare the reference path and actual simulated trajectory.

Random Smooth



Random Sharp



Random Extreme



Evaluation Metrics

Each experiment records:

Root Mean Square Tracking Error (RMSE)

Maximum tracking error

Maximum simulated lateral acceleration

Path progress

Completion status

Numerical failure status

Simulation duration

Results are exported to:

results/random_tracking/random_tracking_results.csv

These scenarios are intended to reveal both controller strengths and failure cases. A visually close trajectory does not necessarily imply acceptable vehicle dynamics or safe operation.

10. Stress Testing and Limitations

Six preliminary stress-test scenarios were evaluated, including:

Normal operation

Large initial lateral offset

Initial heading error

Sharp curve

Low-speed operation

Combined stress conditions

All six passed the project's preliminary tracking and completion criteria.

However, the lateral acceleration target of 0.5 m/s² was exceeded in five of the six scenarios.

The maximum observed lateral accelerations were:

Scenario

Maximum Lateral Acceleration

Normal

1.073 m/s²

Large Offset

6.628 m/s²

Heading Error

0.979 m/s²

Sharp Curve

2.074 m/s²

Low Speed

0.094 m/s²

Combined Stress

6.119 m/s²

These results demonstrate an important limitation: successful path tracking alone does not guarantee that dynamic constraints are satisfied.

The current simulator uses a kinematic model and does not represent tire saturation, detailed suspension behavior, or full vehicle dynamics. It is intended for algorithm development and educational experimentation, not real-world autonomous driving deployment.

11. Installation

Requirements

Python 3.10 or newer

NumPy

Matplotlib

Pillow

Clone the repository:

git clone https://github.com/Xuhaoyuan-creator/autonomous-driving-planning-control.git
cd autonomous-driving-planning-control

Create a virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

12. Running the Experiments

Generate Random Roads

python random_path_generator.py

Run Random Path Tracking

python random_path_tracking.py

Generate Random Path Animation

python random_path_animation.py

Run Scenario Testing

python scenario_testing.py

Optimize Lookahead Gain

python optimize_lookahead.py

Validate Lookahead Performance

python validate_lookahead.py

Run Stress Tests

python stress_testing.py

Generate Performance Report

python generate_performance_report.py

13. Results and Documentation

Generated outputs are organized under the results/ directory.

The project includes:

Tracking trajectory visualizations

Controller comparison plots

Speed and lateral acceleration plots

Optimization and validation results

Stress-test results

Random-road tracking experiments

Animated vehicle simulations

Performance reports

For additional analysis, see:

Performance Report

14. Future Development

Planned improvements include:

Dynamic Constraint Enforcement: Improve speed planning and feedback control to reduce lateral acceleration violations.

More Realistic Vehicle Dynamics: Incorporate steering actuator dynamics, tire behavior, and dynamic bicycle models.

Improved Path Representation: Introduce spline-based paths and arc-length parameterization.

Robustness Evaluation: Expand randomized testing across path curvature, initial conditions, and vehicle parameters.

Controller Comparison: Compare Pure Pursuit against Stanley and Model Predictive Control (MPC).

Automated Testing: Add regression tests and continuous integration for simulation results.

Enhanced Visualization: Create side-by-side animations comparing baseline and optimized controllers.

15. Technologies

Language: Python

Libraries: NumPy, Matplotlib, Pillow

Methods: Kinematic Bicycle Model, Pure Pursuit, PID Control, Curvature-Based Speed Planning, Parameter Sweeps, Simulation-Based Validation

Tools: VS Code, Git, GitHub

Disclaimer

This project is a research-oriented educational simulation. Its results are specific to the implemented models, controllers, parameters, and test conditions.

It does not constitute a validated autonomous driving system and should not be used to control a real vehicle without substantial additional engineering, verification, and safety testing.

Author: Haoyuan Xu

Mechanical Engineering Student, Virginia Tech