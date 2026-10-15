"""
Lunar Explorer AI
-----------------
Controlled mission benchmark:
Original Ice Predictor vs Accessible Ice Predictor.

Same Mission Planner V2, environments, sensor seeds,
measurement limit and travel budget.
"""

import csv
import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.mission_planner import MissionPlanner
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.accessible_ice_predictor import AccessibleIcePredictor

from evaluate_missions import evaluate_observations


SEEDS = range(3000, 3030)

MAX_MEASUREMENTS = 20
MAX_DISTANCE = 2000.0
MAX_SLOPE = 25.0

OUTPUT_FILE = "results/benchmarks/benchmark_ice_predictors_results.csv"


def run_mission(env, predictor, seed):

    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed,
    )

    planner = MissionPlanner(
        env,
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

    return evaluate_observations(env, rover)


def main():

    print("Training original Ice Predictor...")
    original = IcePredictor()
    original.train()

    print("\nTraining Accessible Ice Predictor...")
    accessible = AccessibleIcePredictor()
    accessible.train()

    predictors = [
        ("Original", original),
        ("Accessible", accessible),
    ]

    results = []

    for seed in SEEDS:

        print(f"Running environment {seed}...")

        env = LunarEnvironment(
            size=100,
            extent=500,
            seed=seed,
        ).generate()

        for name, predictor in predictors:

            metrics = run_mission(
                env,
                predictor,
                seed,
            )

            distance = metrics["distance"]
            confirmed = metrics["confirmed"]
            drills = metrics["drill_attempts"]

            results.append({
                "seed": seed,
                "predictor": name,
                **metrics,
                "confirmed_per_100m": (
                    100 * confirmed / distance
                    if distance > 0 else 0.0
                ),
                "confirmed_per_drill": (
                    confirmed / drills
                    if drills > 0 else 0.0
                ),
            })

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("ICE PREDICTOR MISSION BENCHMARK")
    print("=" * 65)

    metrics_to_report = [
        "measurements",
        "detections",
        "drill_attempts",
        "confirmed",
        "distance",
        "confirmed_per_100m",
        "confirmed_per_drill",
    ]

    for name, _ in predictors:

        subset = [
            row for row in results
            if row["predictor"] == name
        ]

        print(f"\n{name}")
        print("-" * 45)

        for metric in metrics_to_report:

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

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
