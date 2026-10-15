"""
Lunar Explorer AI
-----------------
Controlled benchmark: Mission Planner V2 vs V3.

Same predictor, environments, seeds and mission budgets.
"""

import csv
import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.mission_planner import MissionPlanner
from lunar_explorer.mission_planner_v3 import MissionPlannerV3

from evaluate_missions import evaluate_observations


SEEDS = range(5000, 5030)

MAX_MEASUREMENTS = 20
MAX_DISTANCE = 2000.0
MAX_SLOPE = 25.0

OUTPUT_FILE = "results/benchmarks/benchmark_planner_v3_results.csv"


def run_mission(environment, predictor, planner_class, seed):

    rover = LunarRover(
        environment,
        start_position=(50, 50),
        sensor_seed=seed,
    )

    planner = planner_class(
        environment,
        rover,
        predictor,
        max_slope=MAX_SLOPE,
    )

    for _ in range(MAX_MEASUREMENTS):

        observation = planner.explore_step(
            drill=True,
            max_distance=MAX_DISTANCE,
        )

        if observation is None:
            break

    return evaluate_observations(environment, rover)


def main():

    print("Training Ice Predictor...")

    predictor = IcePredictor()
    predictor.train()

    planners = {
        "V2": MissionPlanner,
        "V3": MissionPlannerV3,
    }

    results = []

    for seed in SEEDS:

        print(f"Running environment {seed}...")

        environment = LunarEnvironment(
            size=100,
            extent=500,
            seed=seed,
        ).generate()

        for name, planner_class in planners.items():

            metrics = run_mission(
                environment,
                predictor,
                planner_class,
                seed,
            )

            results.append({
                "seed": seed,
                "planner": name,
                **metrics,
            })

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("MISSION PLANNER V2 VS V3")
    print("=" * 65)

    for name in planners:

        subset = [
            row for row in results
            if row["planner"] == name
        ]

        print(f"\n{name}")
        print("-" * 45)

        for metric in [
            "measurements",
            "detections",
            "drill_attempts",
            "confirmed",
            "distance",
        ]:

            values = np.array(
                [row[metric] for row in subset],
                dtype=float,
            )

            print(
                f"{metric:25s} "
                f"mean={values.mean():8.3f} "
                f"std={values.std():8.3f}"
            )

        total_confirmed = sum(
            row["confirmed"] for row in subset
        )

        total_drills = sum(
            row["drill_attempts"] for row in subset
        )

        total_distance = sum(
            row["distance"] for row in subset
        )

        print(
            f"{'pooled_drill_success':25s} "
            f"{total_confirmed / total_drills if total_drills else 0:.3f}"
        )

        print(
            f"{'pooled_confirmed_per_100m':25s} "
            f"{100 * total_confirmed / total_distance if total_distance else 0:.3f}"
        )

        print(
            f"{'zero_confirmation_fraction':25s} "
            f"{np.mean([row['confirmed'] == 0 for row in subset]):.3f}"
        )

        print(
            f"{'full_measurement_fraction':25s} "
            f"{np.mean([row['measurements'] == MAX_MEASUREMENTS for row in subset]):.3f}"
        )

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
