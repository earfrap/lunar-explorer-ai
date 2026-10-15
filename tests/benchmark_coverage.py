"""
Lunar Explorer AI
Final controlled benchmark: AI V3 vs Systematic exploration.
"""

import csv
import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.rover import LunarRover
from lunar_explorer.mission_planner_v3 import MissionPlannerV3

from evaluate_missions import (
    evaluate_observations,
    run_systematic_mission,
)

from systematic_coverage import run_systematic_coverage

SEEDS = range(7000, 7030)
MAX_MEASUREMENTS = 20
MAX_DISTANCE = 2000.0
MAX_SLOPE = 25.0

OUTPUT_FILE = "results/benchmarks/benchmark_coverage_results.csv"

def run_ai(env, predictor, seed):
    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed,
    )

    planner = MissionPlannerV3(
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


def add_metrics(metrics):
    metrics = metrics.copy()

    distance = metrics["distance"]
    confirmed = metrics["confirmed"]
    drills = metrics["drill_attempts"]

    # Undefined when distance is zero.
    metrics["confirmed_per_100m"] = (
        100.0 * confirmed / distance
        if distance > 0 else float("nan")
    )

    metrics["confirmed_per_drill"] = (
        confirmed / drills
        if drills > 0 else float("nan")
    )

    return metrics


def summarize(results, strategy):
    subset = [
        row for row in results
        if row["strategy"] == strategy
    ]

    print(f"\n{strategy}")
    print("-" * 55)

    for metric in [
        "measurements",
        "detections",
        "confirmed",
        "drill_attempts",
        "distance",
        "confirmed_per_100m",
        "confirmed_per_drill",
    ]:
        values = np.array(
            [row[metric] for row in subset],
            dtype=float,
        )

        valid = values[np.isfinite(values)]

        if len(valid):
            print(
                f"{metric:25s} "
                f"{valid.mean():8.3f} ± {valid.std():8.3f}"
                f"  (n={len(valid)})"
            )
        else:
            print(f"{metric:25s} undefined")

    total_confirmed = sum(row["confirmed"] for row in subset)
    total_distance = sum(row["distance"] for row in subset)
    total_drills = sum(row["drill_attempts"] for row in subset)

    print(
        f"{'pooled_confirmed_per_100m':25s} "
        f"{100 * total_confirmed / total_distance if total_distance else float('nan'):.3f}"
    )
    print(
        f"{'pooled_confirmed_per_drill':25s} "
        f"{total_confirmed / total_drills if total_drills else float('nan'):.3f}"
    )


def paired_comparison(results, metric, reference="Systematic"):
    ai = {
        row["seed"]: row[metric]
        for row in results
        if row["strategy"] == "AI V3"
    }

    systematic = {
        row["seed"]: row[metric]
        for row in results
        if row["strategy"] == reference
    }

    seeds = sorted(set(ai) & set(systematic))

    differences = np.array(
        [ai[s] - systematic[s] for s in seeds],
        dtype=float,
    )

    valid = differences[np.isfinite(differences)]

    print(f"\n{metric}")

    if not len(valid):
        print("No valid paired observations.")
        return

    print(f"Valid pairs:       {len(valid)}")
    print(f"Mean difference:   {valid.mean():+.3f}")
    print(f"Std difference:    {valid.std():.3f}")
    print(f"AI better:         {np.sum(valid > 0)}")
    print(f"Systematic better: {np.sum(valid < 0)}")
    print(f"Ties:              {np.sum(valid == 0)}")

    if len(valid) > 1:
        # Paired bootstrap CI for the mean difference.
        rng = np.random.default_rng(42)
        indices = rng.integers(
            0, len(valid), size=(10000, len(valid))
        )
        bootstrap_means = valid[indices].mean(axis=1)
        low, high = np.percentile(
            bootstrap_means, [2.5, 97.5]
        )

        print(f"Bootstrap 95% CI:  [{low:+.3f}, {high:+.3f}]")


def main():
    print("Training Original IcePredictor...")
    predictor = IcePredictor()
    predictor.train()

    results = []

    for seed in SEEDS:
        print(f"Environment {seed}...")

        env = LunarEnvironment(
            size=100,
            extent=500,
            seed=seed,
        ).generate()

        ai = add_metrics(run_ai(env, predictor, seed))

        systematic = add_metrics(
            run_systematic_mission(
                env,
                predictor,
                seed,
                max_measurements=MAX_MEASUREMENTS,
                max_distance=MAX_DISTANCE,
            )
        )


        coverage_rover = run_systematic_coverage(
            env,
            predictor,
            seed,
            max_measurements=MAX_MEASUREMENTS,
            max_distance=MAX_DISTANCE,
        )

        coverage = add_metrics(
            evaluate_observations(env, coverage_rover)
        )

        results.append({
            "seed": seed,
            "strategy": "AI V3",
            **ai,
        })

        results.append({
            "seed": seed,
            "strategy": "Systematic",
            **systematic,
        })

        results.append({
            "seed": seed,
            "strategy": "Systematic Coverage",
            **coverage,
        })

        print(
            f"  Confirmed: AI={ai['confirmed']} | "
            f"Systematic={systematic['confirmed']} | "
            f"Coverage={coverage['confirmed']}"
        )

    with open(OUTPUT_FILE, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("FINAL BENCHMARK — AI V3 VS SYSTEMATIC")
    print("=" * 65)

    summarize(results, "AI V3")
    summarize(results, "Systematic")
    summarize(results, "Systematic Coverage")


    for reference in ["Systematic", "Systematic Coverage"]:

        print(f"\nPAIRED COMPARISON (AI V3 - {reference})")
        print("-" * 55)

        for metric in [
            "confirmed",
            "confirmed_per_100m",
            "confirmed_per_drill",
            "distance",
            "measurements",
        ]:
            paired_comparison(
                results,
                metric,
                reference=reference,
            )

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
