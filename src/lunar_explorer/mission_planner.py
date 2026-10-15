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
        self.drill_threshold = 0.60
        self.probability_map = (
            predictor.predict_prior_map(environment)
        )

        self.prior_map = self.probability_map.copy()

        self.update_radius = 25.0
        self.update_strength = 0.5

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
                    0.75 * probability
                    + 0.15 * uncertainty
                    - 0.10 * distance / self.environment.size
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

    def update_probability_map(
        self,
        row,
        col,
        observed_probability
    ):

        # Compare sensor-informed prediction
        # with the original prior at this location.

        prior_probability = self.prior_map[row, col]

        correction = (
            observed_probability - prior_probability
        )

        # Work only within a local neighborhood.

        sigma_cells = (
            self.update_radius
            / self.environment.resolution
        )

        influence_cells = int(
            np.ceil(3 * sigma_cells)
        )

        r0 = max(0, row - influence_cells)
        r1 = min(
            self.environment.size,
            row + influence_cells + 1
        )

        c0 = max(0, col - influence_cells)
        c1 = min(
            self.environment.size,
            col + influence_cells + 1
        )

        rows, cols = np.mgrid[r0:r1, c0:c1]

        distance_squared = (
            (rows - row)**2
            + (cols - col)**2
        )

        weights = np.exp(
            -distance_squared
            / (2 * sigma_cells**2)
        )

        # Propagate part of the correction
        # to nearby unvisited locations.

        local_map = self.probability_map[
            r0:r1, c0:c1
        ]

        local_visited = self.visited[
            r0:r1, c0:c1
        ]

        update = (
            self.update_strength
            * correction
            * weights
        )

        local_map[~local_visited] += (
            update[~local_visited]
        )

        np.clip(
            local_map,
            0.0,
            1.0,
            out=local_map
        )

        # At the measured location, use the
        # sensor-informed probability directly.

        self.probability_map[
            row, col
        ] = observed_probability


    def explore_step(self, drill=False, max_distance=None):

        # Select next exploration target.
        target, path = self.select_target()

        if target is None:
            return None

        # Calculate travel distance.
        path_distance = (
            (len(path) - 1)
            * self.environment.resolution
        )

        # Check remaining travel budget.
        if max_distance is not None:

            remaining = (
                max_distance
                - self.rover.distance_travelled
            )

            if path_distance > remaining:
                return None

        # Move rover along the planned path.
        for position in path[1:]:

            self.rover.move_to(*position)

            self.path_history.append(position)

        # Target coordinates.
        row, col = target

        # STEP 1: Acquire non-invasive measurements.
        observation = self.rover.measure(drill=False)

        # STEP 2: Estimate ice probability using
        # temperature, slope, neutron and GPR.
        updated_probability = (
            self.predictor.predict_observation(
                self.environment.temperature[row, col],
                self.environment.slope[row, col],
                observation["neutron"],
                observation["gpr"]
            )
        )

        # Store the sensor-informed probability.
        observation["ice_probability"] = float(
            updated_probability
        )

        # STEP 3: Decide whether to drill.
        should_drill = (
            drill
            and updated_probability >= self.drill_threshold
        )

        observation["drill_attempted"] = bool(
            should_drill
        )

        # STEP 4: Perform selective drilling.
        if should_drill:

            observation["drill"] = (
                self.rover.sensors.drill_measurement(
                    row,
                    col
                )
            )

        # STEP 5: Update nearby ice probabilities.
        self.update_probability_map(
            row,
            col,
            updated_probability
        )

        # Mark location as explored.
        self.visited[row, col] = True

        return observation


