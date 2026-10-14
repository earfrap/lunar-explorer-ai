"""
Lunar Explorer AI
-----------------
Basic rover model.

The rover can:
    - Move to a grid position
    - Acquire synthetic measurements
    - Store observations
"""

from lunar_explorer.sensors import LunarSensors


class LunarRover:

    def __init__(
        self,
        environment,
        start_position=(50, 50),
        sensor_seed=123
    ):

        self.environment = environment

        self.position = start_position

        self.sensors = LunarSensors(
            environment,
            seed=sensor_seed
        )

        self.observations = []

        self.distance_travelled = 0.0

    def move_to(self, row, col):

        if not (
            0 <= row < self.environment.size
            and 0 <= col < self.environment.size
        ):
            raise ValueError("Target outside environment.")

        current_row, current_col = self.position

        distance = (
            (row - current_row)**2
            + (col - current_col)**2
        )**0.5

        distance *= self.environment.resolution

        self.distance_travelled += distance

        self.position = (row, col)

    def measure(self, drill=False):

        row, col = self.position

        observation = {
            "row": row,
            "col": col,
            "neutron": self.sensors.neutron_measurement(
                row, col
            ),
            "gpr": self.sensors.gpr_measurement(
                row, col
            ),
            "drill": None
        }

        if drill:
            observation["drill"] = (
                self.sensors.drill_measurement(
                    row, col
                )
            )

        self.observations.append(observation)

        return observation
