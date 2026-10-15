import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.mission_planner import MissionPlanner


SEEDS = [100, 101, 102]
MAX_MEASUREMENTS = 20
MAX_DISTANCE = 2000.0
ICE_THRESHOLD = 0.35
MAX_DRILL_DEPTH = 1.0


def diagnose(seed, predictor):
    env = LunarEnvironment(
        size=100,
        extent=500,
        seed=seed,
    ).generate()

    rover = LunarRover(
        env,
        start_position=(50, 50),
        sensor_seed=seed,
    )

    planner = MissionPlanner(
        env,
        rover,
        predictor,
        max_slope=25,
    )

    for _ in range(MAX_MEASUREMENTS):
        observation = planner.explore_step(
            drill=True,
            max_distance=MAX_DISTANCE,
        )

        if observation is None:
            break

    accessible_map = (
        (env.ice_content >= ICE_THRESHOLD)
        & (env.ice_depth <= MAX_DRILL_DEPTH)
    )

    visited_accessible = 0
    missed_drills = 0
    confirmed = 0
    false_positives = 0
    drill_attempts = 0

    print(f"\n{'=' * 78}")
    print(f"SEED {seed}")
    print(f"{'=' * 78}")

    print(
        "Step | Position | AI prob | Ice content | Depth | "
        "Drill? | Drill value | Accessible?"
    )

    for step, obs in enumerate(rover.observations, start=1):
        row, col = obs["row"], obs["col"]

        ice = float(env.ice_content[row, col])
        depth = float(env.ice_depth[row, col])

        accessible = bool(accessible_map[row, col])
        drilled = bool(obs.get("drill_attempted", False))
        drill_value = obs.get("drill")

        sensor_positive = (
            drill_value is not None
            and drill_value >= ICE_THRESHOLD
        )

        if accessible:
            visited_accessible += 1

        if accessible and not drilled:
            missed_drills += 1

        if drilled:
            drill_attempts += 1

        if accessible and sensor_positive:
            confirmed += 1

        if sensor_positive and not accessible:
            false_positives += 1

        drill_text = (
            f"{float(drill_value):.3f}"
            if drill_value is not None
            else "-"
        )

        print(
            f"{step:>4} | "
            f"({row:>2},{col:>2}) | "
            f"{obs['ice_probability']:>7.3f} | "
            f"{ice:>11.3f} | "
            f"{depth:>5.3f} | "
            f"{str(drilled):>6} | "
            f"{drill_text:>11} | "
            f"{str(accessible):>11}"
        )

    print("\nSUMMARY")
    print(f"Accessible cells in map: {accessible_map.sum()}")
    print(f"Measurements: {len(rover.observations)}")
    print(f"Accessible cells visited: {visited_accessible}")
    print(f"Accessible cells not drilled: {missed_drills}")
    print(f"Drill attempts: {drill_attempts}")
    print(f"Confirmed: {confirmed}")
    print(f"Positive readings without accessible ice: {false_positives}")
    print(f"Distance travelled: {rover.distance_travelled:.1f} m")


def main():
    predictor = IcePredictor()
    predictor.train()

    for seed in SEEDS:
        diagnose(seed, predictor)


if __name__ == "__main__":
    main()
