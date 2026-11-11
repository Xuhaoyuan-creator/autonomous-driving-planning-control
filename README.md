Autonomous Vehicle Planning & Control Simulation

A Python-based autonomous vehicle simulation project focused on path tracking, motion control, speed planning, and performance optimization.

Developed by Haoyuan Xu | Mechanical Engineering | Virginia Tech

This project implements a modular autonomous vehicle planning and control framework using Python. It combines a kinematic bicycle model, Pure Pursuit lateral control, PID longitudinal control, adaptive lookahead optimization, steering rate limiting, and curvature-based speed planning.

The goal is to develop, test, and improve vehicle control algorithms under different driving conditions, including irregular road geometries and challenging initial vehicle states.

1. Project Demonstration

Random Path Tracking

One of the main features of this project is the ability to track randomly generated, irregular road trajectories.

Unlike simple sinusoidal paths, these roads contain varying curvature, unpredictable turns, and different levels of difficulty.



The animation demonstrates:

Reference trajectory and actual vehicle trajectory

Real-time vehicle position and heading

Adaptive steering control

Vehicle speed and target speed

Path tracking error

Lateral acceleration

Original Vehicle Simulation

The original simulation demonstrates the vehicle following a predefined reference path using Pure Pursuit control.



2. Key Features

Component

Description

Kinematic Bicycle Model

Simulates vehicle position, heading, and velocity

Pure Pursuit Controller

Calculates steering commands for path tracking

Adaptive Lookahead

Adjusts lookahead distance based on vehicle speed

Steering Rate Limiter

Restricts rapid changes in steering angle

PID Speed Controller

Controls longitudinal acceleration and braking

Curvature-Based Speed Planner

Adjusts target speed according to road curvature

Scenario Testing

Evaluates performance across different road geometries

Parameter Optimization

Identifies effective lookahead parameters

Stress Testing

Evaluates controller behavior under challenging conditions

Random Path Generation

Creates reproducible irregular road trajectories

Performance Visualization

Generates trajectory plots, graphs, CSV files, and GIF animations

3. System Overview

The simulation uses a closed-loop control architecture.

Step 1 — Path Generation

A reference trajectory is generated using predefined mathematical paths or randomized control points.

Step 2 — Speed Planning

The speed planner analyzes path curvature and generates a target velocity profile.

Step 3 — Lateral Control

The Pure Pursuit controller calculates the desired steering angle using the vehicle's current position, heading, and lookahead target.

Step 4 — Longitudinal Control

A PID controller calculates acceleration or braking commands to track the target velocity.

Step 5 — Vehicle Dynamics

The kinematic bicycle model updates the vehicle's position, heading, and speed.

Step 6 — Performance Evaluation

The simulation records tracking error, velocity, steering angle, lateral acceleration, and path completion.

These steps repeat throughout the simulation to create a closed-loop vehicle control system.

4. Vehicle Dynamics Model

The project uses a kinematic bicycle model to approximate planar vehicle motion.

The model describes vehicle movement using the following relationships:

x_dot = velocity * cos(yaw)

y_dot = velocity * sin(yaw)

yaw_dot = (velocity / wheelbase) * tan(steering_angle)

velocity_dot = acceleration

Where:

x, y: Vehicle position

velocity: Longitudinal vehicle speed

yaw: Vehicle heading angle

wheelbase: Distance between front and rear axles

steering_angle: Front-wheel steering angle

acceleration: Longitudinal acceleration

This model provides an efficient environment for developing and evaluating vehicle control algorithms.

5. Lateral Control — Pure Pursuit

Pure Pursuit is a geometric path-tracking algorithm that calculates steering commands based on a target point ahead of the vehicle.

The controller uses an adaptive lookahead distance:

lookahead_distance = base_lookahead + gain * velocity

The lookahead distance changes with vehicle speed to balance tracking accuracy and steering responsiveness.

A shorter lookahead distance generally improves responsiveness to curvature changes but can lead to more aggressive steering.

A longer lookahead distance generally produces smoother steering but may increase tracking error in sharp turns.

Steering Rate Limiting

To avoid instantaneous steering changes, a steering rate limiter restricts how quickly the steering command can change between simulation steps.

This improves the realism of steering commands compared with an unrestricted controller, although it does not fully model a physical steering actuator.

6. Lookahead Parameter Optimization

A parameter sweep was conducted to evaluate different adaptive lookahead gains.

Optimization Results

Lookahead Gain (k)

Tracking RMSE

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

The lowest RMSE in this experiment was achieved with:

Optimized Lookahead Gain: k = 0.20

Cross-Scenario Validation

The optimized controller was evaluated across 12 different test conditions.

Results:

Lower RMSE in 12 out of 12 evaluated conditions

Average reported RMSE improvement of 11.11%

Reported spatial path coverage of 100%

These results demonstrate the effectiveness of parameter optimization within the tested simulation conditions.

They do not establish that the selected gain is optimal for every possible road geometry or driving condition.

7. Longitudinal Control and Speed Planning

The project implements a PID longitudinal controller and a curvature-based speed planner.

PID Speed Control

The PID controller calculates acceleration and braking commands based on the difference between target speed and actual vehicle speed.

The controller continuously adjusts acceleration to reduce speed tracking error.

Curvature-Based Speed Planning

The speed planner reduces target speed when approaching sharper turns.

The relationship between vehicle speed, road curvature, and lateral acceleration is:

lateral_acceleration = velocity**2 * abs(curvature)

A corresponding curvature-based speed target can be calculated using:

target_speed = sqrt(max_lateral_acceleration / abs(curvature))

The actual implementation also considers speed and acceleration constraints.

The relevant experiments use a lateral acceleration planning target of:

0.5 m/s²

This value is a planning target, not a strictly enforced limit. Actual lateral acceleration may exceed it during simulation.

8. Scenario Testing

To evaluate performance under different operating conditions, the project includes automated scenario testing.

Road Geometries

Gentle curves

Medium curves

Sharp curves

Test Speeds

6 m/s

8 m/s

10 m/s

12 m/s

Two speed control strategies are evaluated:

Fixed target speed

Curvature-based planned speed

This produces a total of 24 test configurations.

Performance is evaluated using tracking accuracy, speed behavior, path completion, and lateral acceleration.

9. Random Path Tracking Challenge

To move beyond predictable reference trajectories, a random path tracking experiment was developed.

The purpose of this experiment is to evaluate how the controller responds to irregular road geometries.

Random Path Generation

The road generator creates random lateral control points along a forward-moving path.

Interpolation and Gaussian smoothing are applied to create continuous, smooth trajectories suitable for vehicle simulation.

Random seeds make the experiments reproducible.

Three Random Road Scenarios

Scenario

Characteristics

Random Smooth

Gentle, irregular changes in road direction

Random Sharp

More frequent and pronounced curvature changes

Random Extreme

Stronger directional changes and more challenging geometry

Random Smooth



Random Sharp



Random Extreme



Performance Metrics

The random path experiment evaluates:

Root Mean Square Tracking Error (RMSE)

Maximum tracking error

Maximum lateral acceleration

Path progress percentage

Path completion status

Numerical failure status

Simulation duration

Results are exported to:

results/random_tracking/random_tracking_results.csv

The experiment is designed to reveal both successful tracking behavior and controller limitations under more complex road geometries.

10. Stress Testing and Robustness Evaluation

Six stress-test scenarios were used to evaluate controller behavior.

Test Scenario

Description

Normal

Standard initial conditions

Large Offset

Large initial lateral displacement

Heading Error

Incorrect initial heading

Sharp Curve

Increased road curvature

Low Speed

Reduced initial or operating speed

Combined Stress

Multiple challenging conditions combined

All six scenarios passed the project's preliminary tracking and completion criteria.

However, these criteria do not guarantee that the simulated vehicle satisfies all dynamic constraints.

Maximum Lateral Acceleration

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

Five of the six scenarios exceeded the lateral acceleration planning target of 0.5 m/s².

This highlights an important engineering limitation:

Accurate path tracking does not necessarily mean that the vehicle is operating within its desired dynamic constraints.

Improving constraint enforcement is therefore one of the main directions for future development.

11. Installation

Requirements

Python 3.10 or newer

NumPy

Matplotlib

Pillow

Clone the Repository

git clone https://github.com/Xuhaoyuan-creator/autonomous-driving-planning-control.git
cd autonomous-driving-planning-control

Create a Virtual Environment

python -m venv .venv

Activate the Environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install Dependencies

pip install -r requirements.txt

12. Running the Project

Run the following commands from the project root directory.

Generate Random Paths

python random_path_generator.py

Run Random Path Tracking

python random_path_tracking.py

Generate Random Path Animation

python random_path_animation.py

Run Scenario Testing

python scenario_testing.py

Optimize Lookahead Parameters

python optimize_lookahead.py

Validate Lookahead Optimization

python validate_lookahead.py

Run Stress Tests

python stress_testing.py

Generate the Performance Report

python generate_performance_report.py

13. Results and Documentation

Simulation outputs are stored in the results/ directory.

The project generates:

Reference and actual trajectory comparisons

Tracking error plots

Speed tracking plots

Lateral acceleration plots

Lookahead optimization results

Cross-scenario validation results

Stress-testing results

Random path tracking results

Animated vehicle simulations

CSV performance data

Technical Performance Report

View the Full Performance Report

14. Future Development

The next stages of development will focus on improving the realism, reliability, and capability of the simulation.

1. Dynamic Constraint Enforcement

Improve speed planning and feedback control to reduce lateral acceleration violations.

2. Dynamic Bicycle Model

Extend the kinematic model to include more realistic lateral vehicle dynamics.

3. Advanced Path Generation

Introduce spline-based trajectories, improved curvature calculation, and arc-length parameterization.

4. Controller Comparison

Implement and compare additional lateral controllers, including:

Stanley Controller

Model Predictive Control (MPC)

5. Automated Robustness Testing

Expand randomized testing across different road geometries, vehicle parameters, and initial conditions.

6. Improved Visualization

Create side-by-side animations comparing fixed-speed and curvature-based speed planning.

7. Software Engineering Improvements

Add automated regression tests, continuous integration, and improved experiment configuration.

15. Technologies Used

Category

Technologies

Programming

Python

Numerical Computing

NumPy

Visualization

Matplotlib, Pillow

Vehicle Modeling

Kinematic Bicycle Model

Lateral Control

Pure Pursuit

Longitudinal Control

PID

Planning

Curvature-Based Speed Planning

Optimization

Parameter Sweeps and Simulation-Based Evaluation

Development Tools

VS Code, Git, GitHub

Project Limitations

This project uses a simplified kinematic bicycle model and does not fully simulate tire forces, tire saturation, suspension dynamics, or physical steering actuators.

The reported results are specific to the implemented simulation models, parameters, and test scenarios.

This project is intended for educational and engineering experimentation and is not a validated autonomous driving system for deployment on public roads.

About the Developer

Haoyuan Xu

Mechanical Engineering Student
Virginia Tech

Interested in autonomous driving, vehicle dynamics, motion planning, control systems, and simulation-based engineering.

GitHub: Xuhaoyuan-creator