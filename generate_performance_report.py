
from pathlib import Path
from datetime import datetime
import csv
import math
import statistics


# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

REPORT_DIR = RESULTS / "performance_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "PERFORMANCE_REPORT.md"

TARGET_LATERAL_ACCELERATION = 0.5

FILES = {
    "scenario": (
        RESULTS
        / "scenario_testing"
        / "scenario_results.csv"
    ),
    "lookahead": (
        RESULTS
        / "lookahead_optimization"
        / "lookahead_results.csv"
    ),
    "validation": (
        RESULTS
        / "lookahead_validation"
        / "validation_results.csv"
    ),
    "stress": (
        RESULTS
        / "stress_testing"
        / "stress_results.csv"
    ),
}


# ==========================================
# DATA UTILITIES
# ==========================================

def load_csv(path):

    if not path.exists():
        print(f"WARNING: Missing {path}")
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        return list(csv.DictReader(file))


def number(value):

    try:
        result = float(value)

        if math.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return None


def fmt(value, digits=3):

    value = number(value)

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def mean_values(rows, key):

    values = [
        number(row.get(key))
        for row in rows
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return statistics.mean(values)


def markdown_table(headers, rows):

    if not rows:
        return "_No valid data available._\n"

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |"
    ]

    for row in rows:

        lines.append(
            "| " + " | ".join(
                str(value)
                for value in row
            ) + " |"
        )

    return "\n".join(lines) + "\n"


def parse_bool(value):

    return str(value).strip().lower() in (
        "true",
        "1",
        "yes"
    )


# ==========================================
# SECTION 1: PROJECT OVERVIEW
# ==========================================

def build_overview():

    return """
## 1. Project Overview

This project develops a Python-based autonomous
vehicle path tracking and speed control simulation.

The system combines:

- Kinematic bicycle vehicle model
- Pure Pursuit lateral controller
- Adaptive lookahead distance
- Steering rate limiter
- PID longitudinal speed controller
- Curvature-based speed planner
- Automated scenario evaluation
- Robustness stress testing

### Main Objectives

1. Track a reference path accurately.
2. Adjust vehicle speed based on road curvature.
3. Compare different control strategies.
4. Evaluate controller performance across scenarios.
5. Test robustness under challenging conditions.

### Vehicle Control Architecture

```text
Reference Path
    |
    +--> Speed Planner --> PID Controller
    |                         |
    |                         v
    |                    Acceleration
    |
    +--> Pure Pursuit --> Steering Limiter
                              |
                              v
                          Steering

Acceleration + Steering
          |
          v
   Kinematic Bicycle Model
          |
          v
      Vehicle State
```

"""


# ==========================================
# SECTION 2: SCENARIO TESTING
# ==========================================

def build_scenario_section(rows):

    section = """
## 2. Multi-Scenario Testing

The simulation evaluates vehicle behavior
under different road geometries and speeds.

The purpose is to compare control performance
under changing operating conditions.

"""

    if not rows:

        return section + (
            "Scenario testing data was not found.\n\n"
        )

    section += (
        f"Total recorded rows: {len(rows)}\n\n"
    )

    section += (
        "Detailed scenario data is available in "
        "`results/scenario_testing/scenario_results.csv`.\n\n"
    )

    section += (
        "The original scenario evaluation compares "
        "fixed-speed driving with curvature-aware "
        "speed planning. Any claimed improvement "
        "should be based on matched road segments "
        "and verified coverage.\n\n"
    )

    return section


# ==========================================
# SECTION 3: LOOKAHEAD OPTIMIZATION
# ==========================================

def build_lookahead_section(rows):

    section = """
## 3. Adaptive Lookahead Optimization

The Pure Pursuit controller uses:

```text
Ld = 2 + k * v
```

where:

- Ld: lookahead distance (m)
- k: speed-dependent lookahead gain
- v: vehicle speed (m/s)

Different gain values were evaluated to
identify a suitable tracking configuration.

"""

    if not rows:

        return section + (
            "Lookahead optimization data "
            "was not found.\n\n"
        )

    valid = []
    table_rows = []

    for row in rows:

        gain = number(row.get("gain"))
        rmse = number(row.get("path_rmse"))

        if gain is None or rmse is None:
            continue

        valid.append(
            (gain, rmse)
        )

        table_rows.append([
            fmt(gain, 2),
            fmt(rmse),
            fmt(row.get("common_coverage"), 2)
        ])

    section += markdown_table(
        [
            "Gain k",
            "Path RMSE (m)",
            "Common Coverage"
        ],
        table_rows
    )

    if valid:

        best_gain, best_rmse = min(
            valid,
            key=lambda item: item[1]
        )

        section += (
            "\nLowest-RMSE candidate on the "
            "optimization scenario:\n\n"
        )

        section += (
            f"- Gain: **{best_gain:.2f}**\n"
            f"- Path RMSE: **{best_rmse:.3f} m**\n\n"
        )

        section += (
            "This is a scenario-specific candidate, "
            "not proof of global optimality.\n\n"
        )

    return section


# ==========================================
# SECTION 4: CROSS-SCENARIO VALIDATION
# ==========================================

def build_validation_section(rows):

    section = """
## 4. Cross-Scenario Validation

The optimized lookahead gain is compared
against the baseline gain of k = 0.6.

Validation uses matched spatial segments
to reduce bias from different vehicle speeds.

"""

    if not rows:

        return section + (
            "Cross-scenario validation data "
            "was not found.\n\n"
        )

    table_rows = []
    reliable_rows = []

    for row in rows:

        coverage = number(
            row.get("shared_coverage")
        )

        reliable = (
            coverage is not None
            and coverage >= 0.95
        )

        if reliable:
            reliable_rows.append(row)

        table_rows.append([
            row.get("road", "Unknown"),
            fmt(row.get("max_speed"), 1),
            fmt(row.get("baseline_rmse")),
            fmt(row.get("optimized_rmse")),
            fmt(row.get("improvement_percent"), 2),
            fmt(
                coverage * 100
                if coverage is not None
                else None,
                1
            ),
            "Yes" if reliable else "No"
        ])

    section += markdown_table(
        [
            "Road",
            "Speed (m/s)",
            "Baseline RMSE",
            "Optimized RMSE",
            "Improvement (%)",
            "Coverage (%)",
            "Reliable"
        ],
        table_rows
    )

    section += (
        f"\nReliable scenarios: "
        f"**{len(reliable_rows)}/{len(rows)}**\n\n"
    )

    if reliable_rows:

        wins = 0

        for row in reliable_rows:

            baseline = number(
                row.get("baseline_rmse")
            )

            optimized = number(
                row.get("optimized_rmse")
            )

            if (
                baseline is not None
                and optimized is not None
                and optimized < baseline
            ):
                wins += 1

        average_improvement = mean_values(
            reliable_rows,
            "improvement_percent"
        )

        section += (
            f"- Optimized controller wins: "
            f"**{wins}/{len(reliable_rows)}**\n"
        )

        section += (
            "- Mean RMSE improvement: "
            f"**{fmt(average_improvement, 2)}%**\n\n"
        )

        section += (
            "These results describe tracking "
            "accuracy on sufficiently covered "
            "scenarios. Lateral acceleration "
            "and completion should also be "
            "considered before selecting a "
            "final controller.\n\n"
        )

    return section


# ==========================================
# SECTION 5: ROBUSTNESS TESTING
# ==========================================

def build_stress_section(rows):

    section = """
## 5. Robustness Stress Testing

Six conditions are evaluated:

1. Normal driving
2. Large initial lateral offset
3. Initial heading error
4. Sharp curves
5. Low-speed driving
6. Combined disturbances

The tests assess numerical stability,
path tracking error, and route progress.

"""

    if not rows:

        return section + (
            "Stress testing data was not found.\n\n"
        )

    table_rows = []
    passed_count = 0

    for row in rows:

        passed = parse_bool(
            row.get("passed")
        )

        if passed:
            passed_count += 1

        table_rows.append([
            row.get("scenario", "Unknown"),
            fmt(row.get("path_rmse")),
            fmt(row.get("max_path_error")),
            fmt(
                row.get(
                    "max_lateral_acceleration"
                )
            ),
            fmt(
                row.get("completion_percent"),
                1
            ),
            "PASS" if passed else "FAIL"
        ])

    section += markdown_table(
        [
            "Scenario",
            "RMSE (m)",
            "Max Error (m)",
            "Max Lateral Acc (m/s^2)",
            "Completion (%)",
            "Status"
        ],
        table_rows
    )

    section += (
        f"\nStress tests passed: "
        f"**{passed_count}/{len(rows)}**\n\n"
    )

    section += (
        "The PASS status reflects preliminary "
        "tracking and completion criteria. "
        "It does not enforce a strict lateral "
        "acceleration limit or establish "
        "real-world vehicle safety.\n\n"
    )

    section += """
### Interpretation Notes

The test thresholds are preliminary engineering
criteria, not real-world safety certification.

The first 10 seconds are excluded from the
tracking RMSE calculation to allow recovery
from initial disturbances.

Completion is estimated from reference-path
index progress, so the trajectory plot should
also be checked.

Low-speed scenarios may have lower completion
because the simulation has a fixed time limit.

"""

    return section


# ==========================================
# SECTION 6: VISUAL EVIDENCE
# ==========================================

def build_visual_section():

    return """
## 6. Visual Results

### Lookahead Optimization

![Lookahead RMSE](../lookahead_optimization/lookahead_rmse.png)

![Vehicle Trajectories](../lookahead_optimization/lookahead_trajectories.png)

### Cross-Scenario Validation

![Validation RMSE](../lookahead_validation/validation_rmse.png)

![Validation Improvement](../lookahead_validation/validation_improvement.png)

### Robustness Testing

![Stress Trajectories](../stress_testing/stress_trajectories.png)

![Stress RMSE](../stress_testing/stress_rmse.png)

![Stress Completion](../stress_testing/stress_completion.png)

### Vehicle Animation

[View Vehicle Simulation GIF](../animation/vehicle_simulation.gif)

"""


# ==========================================
# SECTION 7: LIMITATIONS
# ==========================================

def build_limitations(stress_rows):

    section = """
## 7. Limitations

The current implementation has several
important limitations:

- Uses a kinematic rather than dynamic
  vehicle model.
- Does not model tire slip or road friction.
- Does not include perception or obstacles.
- Does not include real sensor noise.
- Uses idealized reference paths.
- Has not been validated on a real vehicle.
- Some completion metrics use approximate
  reference-path progress.

### Lateral Acceleration Limitations

The curvature-based speed planner was configured
with a target lateral acceleration of
**0.5 m/s^2**.

However, the robustness tests showed that
actual lateral acceleration can exceed this
target, particularly during recovery from
large initial tracking errors.

"""

    lateral_rows = []
    exceeded = []

    for row in stress_rows:

        value = number(
            row.get("max_lateral_acceleration")
        )

        if value is None:
            continue

        name = row.get(
            "scenario",
            "Unknown"
        )

        lateral_rows.append([
            name,
            fmt(value),
            (
                "Yes"
                if value > TARGET_LATERAL_ACCELERATION
                else "No"
            )
        ])

        if value > TARGET_LATERAL_ACCELERATION:
            exceeded.append(
                (name, value)
            )

    section += markdown_table(
        [
            "Scenario",
            "Max Lateral Acc (m/s^2)",
            "Exceeds Target"
        ],
        lateral_rows
    )

    if lateral_rows:

        section += (
            f"\nScenarios exceeding the "
            f"{TARGET_LATERAL_ACCELERATION:.1f} "
            f"m/s^2 target: "
            f"**{len(exceeded)}/{len(lateral_rows)}**"
            f"\n\n"
        )

    if exceeded:

        worst_name, worst_value = max(
            exceeded,
            key=lambda item: item[1]
        )

        section += (
            f"The highest recorded lateral "
            f"acceleration was "
            f"**{worst_value:.3f} m/s^2** "
            f"in the **{worst_name}** scenario.\n\n"
        )

    section += """
The speed planner calculates target speeds
using reference-path curvature. Actual vehicle
motion can produce greater lateral acceleration,
especially when recovering from initial position
or heading errors.

The current stress-test PASS criteria do not
enforce the lateral acceleration target.

Therefore, passing the tests does not establish
passenger comfort or real-world dynamic safety.

### Future Improvements

1. Monitor and constrain actual lateral acceleration.
2. Reduce speed during large path-tracking errors.
3. Incorporate dynamic vehicle and tire models.
4. Evaluate comfort and stability constraints.
5. Test controller behavior under sensor noise.

The results demonstrate simulated control
performance, not production-level autonomy.

"""

    return section


# ==========================================
# SECTION 8: CONCLUSION
# ==========================================

def build_conclusion():

    return """
## 8. Conclusion

This project implements and evaluates an
autonomous vehicle path tracking and speed
control framework using Python.

The developed system integrates:

- Pure Pursuit lateral control
- PID longitudinal control
- Curvature-aware speed planning
- Steering rate limiting
- Automated performance evaluation
- Parameter optimization
- Cross-scenario validation
- Robustness testing

The experimental results provide a basis
for understanding controller behavior
under different operating conditions.

The robustness evaluation also identifies
limitations in lateral acceleration control,
which should be addressed in future work.

Future development may include dynamic bicycle
models, Model Predictive Control (MPC),
and ROS 2 integration.

"""


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("GENERATING PERFORMANCE REPORT")
    print("=" * 60)

    data = {
        key: load_csv(path)
        for key, path in FILES.items()
    }

    sections = [
        "# Autonomous Vehicle Planning and Control\n",
        "**Technical Performance Report**\n",
        f"Generated: {datetime.now():%Y-%m-%d}\n",
        build_overview(),
        build_scenario_section(
            data["scenario"]
        ),
        build_lookahead_section(
            data["lookahead"]
        ),
        build_validation_section(
            data["validation"]
        ),
        build_stress_section(
            data["stress"]
        ),
        build_visual_section(),
        build_limitations(
            data["stress"]
        ),
        build_conclusion()
    ]

    report = "\n".join(sections)

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print("\nReport generated successfully!")
    print("Location:", REPORT_PATH)

    print("\nData summary:")

    for key, rows in data.items():

        print(
            f"  {key}: {len(rows)} rows"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
