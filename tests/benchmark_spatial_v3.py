"""
Lunar Explorer AI
-----------------
Spatial benchmark: Mission Planner V2 vs V3.

Evaluates the spatial distribution of confirmed drilling
sites across identical synthetic environments.

Ground truth is used exclusively for evaluation.
"""

import csv
import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.mission_planner import MissionPlanner
from lunar_explorer.mission_planner_v3 import MissionPlannerV3
from lunar_explorer.spatial_metrics import evaluate_spatial_confirmations

from evaluate_missions import evaluate_observations


SEEDS = range(5000, 5030)

MAX_MEASUREMENTS = 20
MAX_DISTANCE = 2000.0
MAX_SLOPE = 25.0

SPATIAL_THRESHOLDS = [15.0, 25.0, 50.0]

OUTPUT_FILE = "results/benchmarks/benchmark_spatial_v3_results.csv"


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

    standard = evaluate_observations(environment, rover)

    results = {
        "measurements": standard["measurements"],
        "confirmed": standard["confirmed"],
        "distance": standard["distance"],
    }

    for threshold in SPATIAL_THRESHOLDS:

        spatial = evaluate_spatial_confirmations(
            rover.observations,
            environment,
            distance_threshold=threshold,
        )

        assert (
            spatial["total_confirmed"] == standard["confirmed"]
        ), "Confirmation count mismatch."

        label = int(threshold)

        results[f"groups_{label}m"] = spatial["spatial_groups"]

        results[f"max_separation_{label}m"] = (
            spatial["maximum_separation_m"]
        )

    return results


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

        print(f"Environment {seed}...")

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

    print("\n" + "=" * 70)
    print("SPATIAL BENCHMARK — V2 VS V3")
    print("=" * 70)

    for name in planners:

        subset = [
            row for row in results
            if row["planner"] == name
        ]

        print(f"\n{name}")
        print("-" * 50)

        metrics = [
            "measurements",
            "confirmed",
            "distance",
            "groups_15m",
            "groups_25m",
            "groups_50m",
            "max_separation_25m",
        ]

        for metric in metrics:

            values = np.array(
                [row[metric] for row in subset],
                dtype=float,
            )

            print(
                f"{metric:25s} "
                f"mean={values.mean():8.3f} "
                f"std={values.std():8.3f}"
            )

    print("\n" + "=" * 70)
    print("PAIRED DIFFERENCES (V3 - V2)")
    print("=" * 70)

    for metric in [
        "confirmed",
        "groups_15m",
        "groups_25m",
        "groups_50m",
        "max_separation_25m",
    ]:

        differences = []

        for seed in SEEDS:

            v2 = next(
                row for row in results
                if row["seed"] == seed and row["planner"] == "V2"
            )

            v3 = next(
                row for row in results
                if row["seed"] == seed and row["planner"] == "V3"
            )

            differences.append(v3[metric] - v2[metric])

        differences = np.array(differences)

        print(
            f"{metric:25s} "
            f"mean_delta={differences.mean():8.3f} "
            f"V3_better={np.sum(differences > 0):2d} "
            f"equal={np.sum(differences == 0):2d} "
            f"V2_better={np.sum(differences < 0):2d}"
        )

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
