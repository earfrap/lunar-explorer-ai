"""
Systematic Coverage baseline.

Deterministic spatial coverage with:
- no ML-based destination selection;
- A* navigation;
- shared ML-assisted drilling;
- travel and measurement constraints.
"""

import numpy as np

from lunar_explorer.rover import LunarRover
from lunar_explorer.mission_planner import MissionPlanner


MAX_SLOPE = 25.0



class DummyPredictor:
    def predict_prior_map(self, environment):
        return np.full(
            (environment.size, environment.size),
            0.5,
            dtype=float,
        )

    def predict_prior(self, temperature, slope):
        return 0.5


def run_systematic_coverage(
    env,
    predictor,
    seed,
    max_measurements=20,
    max_distance=2000.0,
):
    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed,
    )

    planner = MissionPlanner(
        env,
        rover,
        DummyPredictor(),
        max_slope=MAX_SLOPE,
    )

    # Deterministic spatial coverage:
    # progressively expanding square grids.
    # The ordering is independent of ice predictions.
    candidates = []
    seen = set()

    for spacing in (20, 10, 5):
        coordinates = list(range(0, env.size, spacing))

        # Serpentine traversal of the grid.
        for i, row in enumerate(coordinates):
            cols = coordinates if i % 2 == 0 else coordinates[::-1]

            for col in cols:
                point = (row, col)

                if point not in seen:
                    candidates.append(point)
                    seen.add(point)

    measurements = 0

    for row, col in candidates:
        if measurements >= max_measurements:
            break

        if env.slope[row, col] > MAX_SLOPE:
            continue

        path = planner.astar(
            rover.position,
            (row, col),
        )

        if path is None:
            continue

        distance = (len(path) - 1) * env.resolution
        if rover.distance_travelled + distance > max_distance:
            continue

        for position in path[1:]:
            rover.move_to(*position)

        observation = rover.measure(drill=False)

        probability = predictor.predict_observation(
            env.temperature[row, col],
            env.slope[row, col],
            observation["neutron"],
            observation["gpr"],
        )

        probability = float(np.asarray(probability).reshape(-1)[0])

        observation["ice_probability"] = probability
        observation["drill_attempted"] = (
            probability >= planner.drill_threshold
        )

        if observation["drill_attempted"]:
            observation["drill"] = rover.sensors.drill_measurement(
                row, col
            )

        measurements += 1

    return rover
