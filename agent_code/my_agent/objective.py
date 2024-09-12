import numpy as np
import networkx as nx


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
    
    def distance_objective(self) -> list:
        """
        Calculates the path to the objective using Dijkstra's algorithm.
        
        :param player_pos: The position of the player.
        :param objective_pos: The position of the objective (coin or crate).
        :param field: The current game field.
        :return: The path to the objective as a list of positions.
        """
        # Create a grid of ones and zeros
        grid = np.zeros(field.shape)
        grid[field == -1] = 2  # Wall (not passable)
        grid[field == -2] = 2  # Crate (not passable)
        grid[field == 0] = 1  # Free space
        grid[field == 1] = 1  # Coin (passable)
        
        # Create a graph from the grid
        graph = nx.grid_2d_graph(grid.shape[0], grid.shape[1])
        
        # Remove nodes that represent walls and crates
        walls_and_crates = np.argwhere(grid == 2)
        for node in walls_and_crates:
            graph.remove_node(tuple(node))
        
        start = tuple(self.player_pos)

        # Find the path to the objective
        try:
            path = nx.shortest_path(graph, start, self.objective), method='dijkstra')
        except nx.NetworkXNoPath:
            # No path found to this objective
            path = []

        return path


