"""
Lunar Explorer AI
-----------------
Accessible ice prediction.

Experimental variant of IcePredictor.

Positive target:
    ice_content >= 0.35
    AND
    ice_depth <= 1.0 m

Depth is used only to construct training labels.
It is NOT provided as an input feature to the models.
"""

import numpy as np

from lunar_explorer.environment import LunarEnvironment
from lunar_explorer.sensors import LunarSensors
from lunar_explorer.ice_predictor import IcePredictor


class AccessibleIcePredictor(IcePredictor):

    def __init__(
        self,
        ice_threshold=0.35,
        max_depth=1.0,
        random_state=42,
    ):

        super().__init__(
            ice_threshold=ice_threshold,
            random_state=random_state,
        )

        self.max_depth = max_depth

    def generate_training_data(
        self,
        n_environments=20,
        samples_per_environment=500,
        seed=1000,
    ):

        rng = np.random.default_rng(seed)

        X_prior = []
        X_observation = []
        y = []

        for i in range(n_environments):

            env = LunarEnvironment(
                size=100,
                extent=500,
                seed=seed + i,
            ).generate()

            sensors = LunarSensors(
                env,
                seed=seed + 10000 + i,
            )

            rows = rng.integers(
                0,
                env.size,
                samples_per_environment,
            )

            cols = rng.integers(
                0,
                env.size,
                samples_per_environment,
            )

            for row, col in zip(rows, cols):

                temperature = env.temperature[row, col]
                slope = env.slope[row, col]

                neutron = sensors.neutron_measurement(
                    row, col
                )

                gpr = sensors.gpr_measurement(
                    row, col
                )

                ice = env.ice_content[row, col]
                depth = env.ice_depth[row, col]

                X_prior.append([
                    temperature,
                    slope,
                ])

                X_observation.append([
                    temperature,
                    slope,
                    neutron,
                    gpr,
                ])

                y.append(
                    int(
                        ice >= self.ice_threshold
                        and depth <= self.max_depth
                    )
                )

        return (
            np.asarray(X_prior),
            np.asarray(X_observation),
            np.asarray(y),
        )
