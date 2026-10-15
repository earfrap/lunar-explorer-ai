"""
Basic functional test for Mission Planner V3.
"""

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.rover import LunarRover
from lunar_explorer.ice_predictor import IcePredictor
from lunar_explorer.mission_planner_v3 import MissionPlannerV3


def main():

    environment = LunarEnvironment(
        size=100,
        extent=500,
        seed=5000
    ).generate()

    predictor = IcePredictor()
    predictor.train()

    rover = LunarRover(
        environment,
        start_position=(50, 50),
        sensor_seed=5000
    )

    planner = MissionPlannerV3(
        environment,
        rover,
        predictor,
        max_slope=25.0
    )

    max_distance = 200.0

    for step in range(20):

        observation = planner.explore_step(
            drill=True,
            max_distance=max_distance
        )

        if observation is None:
            print(
                f"Mission stopped at step {step + 1}"
            )
            break

        print(
            f"Step {step + 1:02d} | "
            f"Position: {rover.position} | "
            f"Distance: {rover.distance_travelled:.1f} m"
        )

        assert (
            rover.distance_travelled
            <= max_distance + 1e-6
        ), "Travel budget exceeded!"

    print("\nV3 functional test completed.")
    print(
        f"Final distance: "
        f"{rover.distance_travelled:.1f} m"
    )


if __name__ == "__main__":
    main()
