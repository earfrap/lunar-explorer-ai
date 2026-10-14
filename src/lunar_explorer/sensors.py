"""
Lunar Explorer AI
-----------------
Simplified synthetic geophysical sensors.

Sensors:
    - Neutron detector
    - Ground-penetrating radar
    - Drill

These are conceptual sensor-response models,
not calibrated representations of real instruments.
"""

import numpy as np


class LunarSensors:

    def __init__(self, environment, seed=123):

        self.environment = environment
        self.rng = np.random.default_rng(seed)

    def neutron_measurement(self, row, col):

        env = self.environment

        # Spatially averaged hydrogen-related signal
        r0 = max(0, row - 2)
        r1 = min(env.size, row + 3)

        c0 = max(0, col - 2)
        c1 = min(env.size, col + 3)

        ice = env.ice_content[r0:r1, c0:c1]
        depth = env.ice_depth[r0:r1, c0:c1]

        attenuation = np.exp(-depth / 0.8)

        signal = np.mean(ice * attenuation)

        noise = self.rng.normal(0, 0.04)

        return float(np.clip(signal + noise, 0, 1))

    def gpr_measurement(self, row, col):

        env = self.environment

        ice = env.ice_content[row, col]
        depth = env.ice_depth[row, col]

        # Simplified radar response
        signal = (
            0.6 * ice
            + 0.4 * np.exp(-depth / 1.5)
        )

        noise = self.rng.normal(0, 0.06)

        return float(np.clip(signal + noise, 0, 1))

    def drill_measurement(self, row, col):

        env = self.environment

        ice = env.ice_content[row, col]
        depth = env.ice_depth[row, col]

        max_drill_depth = 1.0

        if depth > max_drill_depth:
            return 0.0

        noise = self.rng.normal(0, 0.02)

        return float(np.clip(ice + noise, 0, 1))
