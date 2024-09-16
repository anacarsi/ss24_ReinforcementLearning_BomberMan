import numpy as np
import networkx as nx
from typing import Tuple, List
import heapq


class Objective:
    def __init__(self, task: int, game_state: dict):
        self.task = task
        self.field = game_state['field']
        self.coin_array = game_state['coins']
        self.crates = np.argwhere(self.field == -2)
        self.player_pos = game_state['self'][3]
        self.objective = self.set_objective()

    def set_objective(self) -> tuple:
        """
        Determines the objective based on the current game state and the specified task.
        
        :param game_state: The dictionary containing the current state of the game.
        :param task: The task to focus on (1 for collecting coins only, 2 for dealing with crates).
        :return: The position of the objective.
        """

        if self.task == 1:
            # Task 1: Collect coins as quickly as possible
            if self.coin_array:
                objective = self.coin_array[0]  # Nearest coin (assuming they are provided in order of proximity)
            else:
                # No coins to collect
                objective = self.player_pos
                print("No coins left to collect.")

        elif self.task == 2:
            # Task 2: Deal with crates and collect hidden coins
            if self.coin_array:
                objective = self.coin_array[0]  # Nearest coin
            elif self.crates.size > 0:
                # No coins, so find the nearest crate
                objective = self.crates[0]
            else:
                # No coins or crates
                objective = self.player_pos
                print("No coins left to collect, no crates to exploit.")

        else:
            raise ValueError("Invalid task number. Please specify either task 1 or 2.")

        return tuple(objective)

    @staticmethod
    def distance_objective(start_pos: Tuple[int, int], objective_pos: Tuple[int, int], field: np.ndarray) -> List[Tuple[int, int]]:
        """
        Calculate the shortest path from start_pos to objective_pos using Dijkstra's algorithm.

        :param start_pos: Starting position (player's position).
        :param objective_pos: Target position (objective's position).
        :param field: The game field (grid) representation.
        :return: List of coordinates representing the shortest path from start_pos to objective_pos.
        """
        rows, cols = field.shape
        start_x, start_y = start_pos
        goal_x, goal_y = objective_pos
        
        # Define movements (right, left, down, up)
        movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Initialize distances and priority queue
        distances = np.full((rows, cols), np.inf)
        distances[start_x, start_y] = 0
        priority_queue = [(0, start_pos)]
        came_from = {start_pos: None}

        while priority_queue:
            current_distance, current_pos = heapq.heappop(priority_queue)
            current_x, current_y = current_pos

            # Early exit if we reach the goal
            if current_pos == (goal_x, goal_y):
                break

            # Explore neighbors
            for move_x, move_y in movements:
                neighbor_x, neighbor_y = current_x + move_x, current_y + move_y

                if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                    new_distance = current_distance + 1
                    if new_distance < distances[neighbor_x, neighbor_y]:
                        distances[neighbor_x, neighbor_y] = new_distance
                        heapq.heappush(priority_queue, (new_distance, (neighbor_x, neighbor_y)))
                        came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)

        # Reconstruct path
        path = []
        step = (goal_x, goal_y)
        while step is not None:
            path.append(step)
            step = came_from.get(step, None)
        path.reverse()
        
        return path




