"""
Lunar Explorer AI
-----------------
Spatial analysis of confirmed drilling sites.

Metrics:
    - Total confirmed sites
    - Spatially distinct confirmation groups
    - Maximum separation between confirmed sites

The grouping distance is an operational parameter,
not a geological deposit-size estimate.
"""

import numpy as np
from lunar_explorer.scientific_metrics import (
    is_validated_confirmation,
)



def get_confirmed_positions(
    observations,
    environment,
    drill_threshold=0.35,
):
    """
    Extract validated drilling positions.

    Ground truth is used exclusively for
    post-hoc scientific evaluation.
    """

    positions = []

    for observation in observations:

        if is_validated_confirmation(
            environment,
            observation,
        ):
            positions.append(
                (
                    int(observation["row"]),
                    int(observation["col"]),
                )
            )

    return positions


def count_spatial_groups(
    positions,
    resolution,
    distance_threshold=25.0,
):
    """
    Count connected groups of confirmed positions.

    Two positions are connected if their physical
    separation is <= distance_threshold.
    """

    n = len(positions)

    if n == 0:
        return 0

    visited = set()
    groups = 0

    for start in range(n):

        if start in visited:
            continue

        groups += 1

        stack = [start]
        visited.add(start)

        while stack:

            i = stack.pop()

            for j in range(n):

                if j in visited:
                    continue

                distance = (
                    np.hypot(
                        positions[i][0] - positions[j][0],
                        positions[i][1] - positions[j][1],
                    )
                    * resolution
                )

                if distance <= distance_threshold:

                    visited.add(j)
                    stack.append(j)

    return groups


def maximum_confirmation_separation(
    positions,
    resolution,
):
    """
    Maximum Euclidean separation between confirmed sites.
    """

    if len(positions) < 2:
        return 0.0

    maximum = 0.0

    for i in range(len(positions)):

        for j in range(i + 1, len(positions)):

            distance = (
                np.hypot(
                    positions[i][0] - positions[j][0],
                    positions[i][1] - positions[j][1],
                )
                * resolution
            )

            maximum = max(maximum, distance)

    return float(maximum)

def evaluate_spatial_confirmations(
    observations,
    environment,
    distance_threshold=25.0,
    drill_threshold=0.35,
):
    """
    Compute spatial metrics using the same
    confirmation definition as evaluate_observations().
    """

    positions = get_confirmed_positions(
        observations,
        environment,
        drill_threshold=drill_threshold,
    )

    return {
        "total_confirmed": len(positions),

        "spatial_groups": count_spatial_groups(
            positions,
            environment.resolution,
            distance_threshold,
        ),

        "maximum_separation_m":
            maximum_confirmation_separation(
                positions,
                environment.resolution,
            ),
    }
