"""
Lunar Explorer AI
-----------------
Compare AI-guided and systematic exploration.
"""

import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.mission_planner import MissionPlanner


ICE_THRESHOLD = 0.35
MAX_DISTANCE = 1000.0
MAX_MEASUREMENTS = 20
MAX_SLOPE = 25.0


def evaluate_observations(env, rover):

    detections = 0
    confirmed = 0

    for obs in rover.observations:

        row = obs["row"]
        col = obs["col"]

        ice = env.ice_content[row, col]
        depth = env.ice_depth[row, col]

        if ice >= ICE_THRESHOLD:
            detections += 1

        if (
            ice >= ICE_THRESHOLD
            and depth <= 1.0
            and obs["drill"] is not None
            and obs["drill"] >= ICE_THRESHOLD
        ):
            confirmed += 1

    n = len(rover.observations)

    distance = rover.distance_travelled

    return {
        "measurements": n,
        "detections": detections,
        "confirmed": confirmed,
        "detection_rate": (
            detections / n if n else 0.0
        ),
        "distance": distance,
        "detections_per_100m": (
            100 * detections / distance
            if distance > 0 else 0.0
        )
    }


def run_ai_mission(env, predictor, seed):

    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed
    )

    planner = MissionPlanner(
        env,
        rover,
        predictor,
        max_slope=MAX_SLOPE
    )

    for _ in range(MAX_MEASUREMENTS):

        observation = planner.explore_step(
            drill=True,
            max_distance=MAX_DISTANCE
        )

        if observation is None:
            break

    return evaluate_observations(env, rover)


def run_systematic_mission(env, seed):

    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed
    )

    # Use the same planner for A* mobility constraints.
    # Target selection is independent of AI probabilities.
    class DummyPredictor:

        def predict_prior_map(self, environment):
            return np.full(
                (environment.size, environment.size),
                0.5
            )

    planner = MissionPlanner(
        env,
        rover,
        DummyPredictor(),
        max_slope=MAX_SLOPE
    )

    # Systematic exploration around the landing site.
    # Fixed spatial pattern, independent of ice probability.

    start_row, start_col = rover.position

    coordinates = []

    for radius in range(1, env.size):

        ring = []

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):

                if max(abs(dr), abs(dc)) != radius:
                    continue

                row = start_row + dr
                col = start_col + dc

                if (
                    0 <= row < env.size
                    and 0 <= col < env.size
                ):
                    ring.append((row, col))

        ring.sort()

        if radius % 2 == 0:
            ring.reverse()

        coordinates.extend(ring)

    for target in coordinates:

        if len(rover.observations) >= MAX_MEASUREMENTS:
            break

        path = planner.astar(
            rover.position, target
        )

        if path is None:
            continue

        path_distance = (
            (len(path) - 1)
            * env.resolution
        )

        if (
            rover.distance_travelled + path_distance
            > MAX_DISTANCE
        ):
            continue

        for row, col in path[1:]:
            rover.move_to(row, col)

        rover.measure(drill=True)

    return evaluate_observations(env, rover)


def main():

    print("Training AI model...")

    predictor = IcePredictor()
    predictor.train()

    results = []

    for seed in range(100, 110):

        env = LunarEnvironment(
            seed=seed
        ).generate()

        ai = run_ai_mission(
            env, predictor, seed
        )

        systematic = run_systematic_mission(
            env, seed
        )

        results.append({
            "seed": seed,
            "AI": ai,
            "Systematic": systematic
        })

        print(
            f"Seed {seed}: "
            f"AI={ai['detections']} detections, "
            f"Systematic={systematic['detections']}"
        )

    print("\nAVERAGE RESULTS (10 environments)")
    print("-" * 65)

    metrics = [
        "measurements",
        "detections",
        "confirmed",
        "detection_rate",
        "distance",
        "detections_per_100m"
    ]

    for metric in metrics:

        ai_mean = np.mean([
            r["AI"][metric]
            for r in results
        ])

        sys_mean = np.mean([
            r["Systematic"][metric]
            for r in results
        ])

        print(
            f"{metric:22s} "
            f"AI: {ai_mean:8.3f} | "
            f"Systematic: {sys_mean:8.3f}"
        )


if __name__ == "__main__":
    main()
