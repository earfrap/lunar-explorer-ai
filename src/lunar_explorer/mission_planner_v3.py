"""
Lunar Explorer AI
-----------------
Mission Planner V3.

Experimental improvement:
Budget-aware target selection.

Unlike V2, V3 searches for the highest-scoring
reachable target within the remaining travel budget.

All prediction, drilling and probability-update
mechanisms are inherited from V2.
"""

from collections import deque

import numpy as np

from lunar_explorer.mission_planner import MissionPlanner


class MissionPlannerV3(MissionPlanner):

    def select_target(self, max_distance=None):

        if max_distance is None:
            max_distance = getattr(
                self,
                "_active_max_distance",
                None
            )

        current = self.rover.position
        if max_distance is None:
            remaining = float("inf")
        else:
            remaining = (
                max_distance
                - self.rover.distance_travelled
            )

        if remaining < 0:
            return None, None

        resolution = self.environment.resolution

        if resolution <= 0:
            raise ValueError(
                "Environment resolution must be positive."
            )

        max_steps = int(
            np.floor(remaining / resolution + 1e-9)
        ) if np.isfinite(remaining) else float("inf")

        # BFS: shortest paths on an unweighted grid.
        queue = deque([current])

        distances = {current: 0}
        parents = {}

        while queue:

            position = queue.popleft()

            steps = distances[position]

            if steps >= max_steps:
                continue

            for neighbor in self.neighbors(position):

                if neighbor in distances:
                    continue

                distances[neighbor] = steps + 1
                parents[neighbor] = position

                queue.append(neighbor)

        best_target = None
        best_score = -float("inf")

        for (row, col), steps in distances.items():

            if self.visited[row, col]:
                continue

            if (
                self.environment.slope[row, col]
                > self.max_slope
            ):
                continue

            probability = self.probability_map[row, col]

            uncertainty = (
                1.0 - abs(2.0 * probability - 1.0)
            )

            euclidean_distance = np.hypot(
                row - current[0],
                col - current[1]
            )

            # Same scoring function as V2.
            score = (
                0.75 * probability
                + 0.15 * uncertainty
                - 0.10
                * euclidean_distance
                / self.environment.size
            )

            if score > best_score:

                best_score = score
                best_target = (row, col)

        if best_target is None:
            return None, None

        # Reconstruct shortest path from BFS parents.
        path = [best_target]

        position = best_target

        while position != current:

            position = parents[position]
            path.append(position)

        path.reverse()

        return best_target, path



    def explore_step(self, drill=False, max_distance=None):
        """
        Execute one exploration step using budget-aware
        target selection.

        The exploration workflow remains inherited from V2.
        """

        if max_distance is not None:
            remaining = (
                max_distance
                - self.rover.distance_travelled
            )

            if remaining < 0:
                return None

        # Store the budget for the current exploration step.
        self._active_max_distance = max_distance

        try:
            return super().explore_step(
                drill=drill,
                max_distance=max_distance
            )
        finally:
            self._active_max_distance = None


