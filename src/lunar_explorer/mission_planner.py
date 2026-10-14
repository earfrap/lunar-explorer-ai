"""
Lunar Explorer AI
-----------------
Autonomous rover exploration.

The planner:
    - Maintains an ice probability map
    - Selects promising exploration targets
    - Uses A* pathfinding
    - Avoids steep terrain
    - Updates predictions after measurements
"""

import heapq
import numpy as np


class MissionPlanner:

    def __init__(
        self,
        environment,
        rover,
        predictor,
        max_slope=25.0
    ):

        self.environment = environment
        self.rover = rover
        self.predictor = predictor

        self.max_slope = max_slope

        self.probability_map = (
            predictor.predict_prior_map(environment)
        )

        self.visited = np.zeros(
            (environment.size, environment.size),
            dtype=bool
        )

        self.path_history = [rover.position]

    def neighbors(self, position):

        row, col = position

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        for dr, dc in directions:

            nr = row + dr
            nc = col + dc

            if (
                0 <= nr < self.environment.size
                and 0 <= nc < self.environment.size
                and self.environment.slope[nr, nc]
                    <= self.max_slope
            ):
                yield (nr, nc)

    def heuristic(self, a, b):

        return (
            abs(a[0] - b[0])
            + abs(a[1] - b[1])
        )

    def astar(self, start, goal):

        queue = [(0, start)]

        came_from = {}

        g_score = {start: 0}

        while queue:

            _, current = heapq.heappop(queue)

            if current == goal:

                path = [current]

                while current in came_from:
                    current = came_from[current]
                    path.append(current)

                return path[::-1]

            for neighbor in self.neighbors(current):

                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(
                    neighbor, float("inf")
                ):

                    came_from[neighbor] = current

                    g_score[neighbor] = tentative_g

                    priority = (
                        tentative_g
                        + self.heuristic(neighbor, goal)
                    )

                    heapq.heappush(
                        queue,
                        (priority, neighbor)
                    )

        return None

    def select_target(self):

        current = self.rover.position

        candidates = []

        for row in range(self.environment.size):

            for col in range(self.environment.size):

                if self.visited[row, col]:
                    continue

                if (
                    self.environment.slope[row, col]
                    > self.max_slope
                ):
                    continue

                probability = self.probability_map[
                    row, col
                ]

                uncertainty = (
                    1.0 - abs(2.0 * probability - 1.0)
                )

                distance = np.hypot(
                    row - current[0],
                    col - current[1]
                )

                score = (
                    0.6 * probability
                    + 0.3 * uncertainty
                    - 0.1 * distance / self.environment.size
                )

                candidates.append(
                    (score, (row, col))
                )

        candidates.sort(reverse=True)

        for _, target in candidates[:100]:

            path = self.astar(
                current, target
            )

            if path is not None:
                return target, path

        return None, None
    def explore_step(self, drill=False, max_distance=None):

        target, path = self.select_target()

        if target is None:
            return None

        path_distance = (
            (len(path) - 1)
            * self.environment.resolution
        )

        if max_distance is not None:

            remaining = (
                max_distance
                - self.rover.distance_travelled
            )

            if path_distance > remaining:
                return None

        for row, col in path[1:]:

            self.rover.move_to(row, col)

            self.path_history.append(
                (row, col)
            )

        observation = self.rover.measure(
            drill=drill
        )

        row, col = target

        updated_probability = (
            self.predictor.predict_observation(
                self.environment.temperature[row, col],
                self.environment.slope[row, col],
                observation["neutron"],
                observation["gpr"]
            )
        )

        self.probability_map[
            row, col
        ] = updated_probability

        self.visited[row, col] = True

        return observation

