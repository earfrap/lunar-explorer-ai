"""
Lunar Explorer AI
-----------------
Synthetic lunar environment generator.

Generates:
    - Lunar topography
    - Surface slope
    - Synthetic surface temperature
    - Hidden subsurface ice distribution

All physical relationships are simplified and intended
for autonomous exploration simulations.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


class LunarEnvironment:

    def __init__(self, size=100, extent=500, seed=42):

        self.size = size
        self.extent = extent
        self.resolution = extent / size

        self.rng = np.random.default_rng(seed)

        self.x = np.linspace(
            0, extent, size, endpoint=False
        ) + self.resolution / 2

        self.y = self.x.copy()

        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.topography = None
        self.slope = None
        self.temperature = None
        self.ice_content = None
        self.ice_depth = None
    def generate_topography(self, n_craters=12):

        terrain = self.rng.normal(
            0, 1, (self.size, self.size)
        )

        terrain = gaussian_filter(terrain, sigma=8)

        terrain = terrain / max(terrain.std(), 1e-8)
        terrain *= 3.0

        for _ in range(n_craters):

            cx = self.rng.uniform(0, self.extent)
            cy = self.rng.uniform(0, self.extent)

            radius = self.rng.uniform(15, 55)
            depth = self.rng.uniform(3, 12)

            distance = np.sqrt(
                (self.X - cx)**2 +
                (self.Y - cy)**2
            )

            depression = -depth * np.exp(
                -(distance / (0.65 * radius))**4
            )

            rim = 0.2 * depth * np.exp(
                -((distance - radius) /
                  (0.2 * radius))**2
            )

            terrain += depression + rim

        self.topography = terrain

        return terrain

    def calculate_slope(self):

        if self.topography is None:
            raise ValueError("Generate topography first.")

        dy, dx = np.gradient(
            self.topography,
            self.resolution
        )

        gradient = np.sqrt(dx**2 + dy**2)

        self.slope = np.degrees(
            np.arctan(gradient)
        )

        return self.slope

    def generate_temperature(self):

        if self.topography is None:
            raise ValueError("Generate topography first.")

        if self.slope is None:
            self.calculate_slope()

        elevation_effect = (
            self.topography -
            self.topography.mean()
        )

        temperature = (
            105
            + 0.15 * elevation_effect
            - 0.4 * self.slope
        )

        spatial_noise = self.rng.normal(
            0, 1, (self.size, self.size)
        )

        spatial_noise = gaussian_filter(
            spatial_noise, sigma=6
        )

        spatial_noise /= max(
            spatial_noise.std(), 1e-8
        )

        temperature += 5 * spatial_noise

        self.temperature = np.clip(
            temperature, 35, 140
        )

        return self.temperature

    def generate_ice(self):

        if self.temperature is None:
            self.generate_temperature()

        temperature_factor = np.clip(
            (110 - self.temperature) / 60,
            0, 1
        )

        random_field = self.rng.normal(
            0, 1, (self.size, self.size)
        )

        random_field = gaussian_filter(
            random_field, sigma=7
        )

        random_field -= random_field.min()

        random_field /= max(
            np.ptp(random_field), 1e-8
        )

        ice = (
            0.65 * temperature_factor
            + 0.35 * random_field
        )

        self.ice_content = np.clip(
            ice, 0, 1
        )

        return self.ice_content


    def generate_ice_depth(self):

        if self.ice_content is None:
            self.generate_ice()

        random_field = self.rng.normal(
            0, 1, (self.size, self.size)
        )

        random_field = gaussian_filter(
            random_field, sigma=10
        )

        random_field -= random_field.min()

        random_field /= max(
            np.ptp(random_field), 1e-8
        )

        self.ice_depth = (
            0.1 + 1.9 * random_field
        )

        return self.ice_depth


    def generate(self):

        self.generate_topography()
        self.calculate_slope()
        self.generate_temperature()
        self.generate_ice()
        self.generate_ice_depth()

        return self


    def plot_environment(self):

        if self.ice_content is None:
            raise ValueError("Generate environment first.")

        fig, axes = plt.subplots(
            2, 2, figsize=(12, 10)
        )

        maps = [
            (self.topography, "Topography", "terrain", "m"),
            (self.slope, "Slope", "inferno", "degrees"),
            (self.temperature, "Temperature", "coolwarm", "K"),
            (self.ice_content, "Subsurface Ice", "Blues", "relative index")
        ]

        for ax, (data, title, cmap, unit) in zip(
            axes.flat, maps
        ):

            image = ax.imshow(
                data,
                origin="lower",
                extent=[0, self.extent, 0, self.extent],
                cmap=cmap
            )

            ax.set_title(title)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            fig.colorbar(
                image,
                ax=ax,
                label=unit
            )

        fig.suptitle(
            "Lunar Explorer AI — Synthetic Environment",
            fontsize=15
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    environment = LunarEnvironment(
        size=100,
        extent=500,
        seed=42
    )

    environment.generate()

    environment.plot_environment()
