"""
Lunar Explorer AI — Controlled Planner Benchmark

Compare:
1. Original AI planner
2. Modified AI planner (V2)
3. Systematic exploration with AI-assisted drilling

Same environments, model, measurement limit and travel budget.
"""

import csv
import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.mission_planner import MissionPlanner as PlannerV2
from lunar_explorer.mission_planner_baseline import (
    MissionPlanner as PlannerBaseline
)

from evaluate_missions import (
    evaluate_observations,
    run_systematic_mission,
)


SEEDS = range(100, 130)
MAX_MEASUREMENTS = 20
MAX_DISTANCE = 2000.0
MAX_SLOPE = 25.0

OUTPUT_FILE = "results/benchmarks/benchmark_planners_results.csv"


def run_ai(env, predictor, seed, planner_class):

    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed,
    )

    planner = planner_class(
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


def run_benchmark():

    predictor = IcePredictor()
    predictor.train()

    results = []

    strategies = [
        ("AI Baseline", PlannerBaseline),
        ("AI V2", PlannerV2),
        ("Systematic + AI drilling", None),
    ]

    for seed in SEEDS:

        print(f"Running seed {seed}...")

        env = LunarEnvironment(
            size=100,
            extent=500,
            seed=seed,
        ).generate()

        for strategy_name, planner_class in strategies:

            if planner_class is None:

                metrics = run_systematic_mission(
                    env,
                    predictor,
                    seed,
                    max_measurements=MAX_MEASUREMENTS,
                    max_distance=MAX_DISTANCE,
                )

            else:

                metrics = run_ai(
                    env,
                    predictor,
                    seed,
                    planner_class,
                )

            distance = metrics["distance"]
            confirmed = metrics["confirmed"]
            drill_attempts = metrics["drill_attempts"]

            metrics["confirmed_per_100m"] = (
                100.0 * confirmed / distance
                if distance > 0 else 0.0
            )

            metrics["confirmed_per_drill"] = (
                confirmed / drill_attempts
                if drill_attempts > 0 else 0.0
            )

            results.append({
                "seed": seed,
                "strategy": strategy_name,
                **metrics,
            })

    return results


def save_results(results):

    fieldnames = list(results[0].keys())

    with open(OUTPUT_FILE, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def print_summary(results):

    strategies = sorted(
        set(row["strategy"] for row in results)
    )

    metrics = [
        "measurements",
        "detections",
        "drill_attempts",
        "confirmed",
        "distance",
        "confirmed_per_100m",
        "confirmed_per_drill",
    ]

    print("\n" + "=" * 76)
    print("BENCHMARK SUMMARY — 30 SEEDS")
    print("=" * 76)

    for strategy in strategies:

        subset = [
            row for row in results
            if row["strategy"] == strategy
        ]

        print(f"\n{strategy}")
        print("-" * 50)

        for metric in metrics:

            values = np.array(
                [row[metric] for row in subset],
                dtype=float,
            )

            print(
                f"{metric:25s} "
                f"mean={np.mean(values):8.3f} "
                f"std={np.std(values):8.3f}"
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

        pooled_drill_success = (
            total_confirmed / total_drills
            if total_drills else 0.0
        )

        pooled_distance_efficiency = (
            100.0 * total_confirmed / total_distance
            if total_distance else 0.0
        )

        zero_confirmation_fraction = np.mean([
            row["confirmed"] == 0
            for row in subset
        ])

        print(
            f"{'pooled_drill_success':25s} "
            f"{pooled_drill_success:8.3f}"
        )

        print(
            f"{'pooled_confirmed_per_100m':25s} "
            f"{pooled_distance_efficiency:8.3f}"
        )

        print(
            f"{'zero_confirmation_fraction':25s} "
            f"{zero_confirmation_fraction:8.3f}"
        )


if __name__ == "__main__":

    results = run_benchmark()

    save_results(results)

    print_summary(results)

    print(f"\nResults saved to {OUTPUT_FILE}")
