import os
import pickle
import random
import networkx as nx
from networkx import grid_2d_graph, shortest_path
import numpy as np
import json


#ACTIONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'WAIT', 'BOMB']
ACTIONS = [ 'UP', 'DOWN', 'RIGHT', 'LEFT', 'WAIT']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-training-set.csv"):
        self.logger.info("Setting up model from scratch.")
        self.train_set = []
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-training-set.csv", "r") as file:
            self.model = json.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    max_steps = 2
    epsilon = 0.5
    #move to valid spaces

    field = np.transpose(game_state['field'])
    field[-1, :] = -100
    player = game_state['self']
    player_pos = player[3]
    player_x,player_y = player_pos

    #up,down, left, right
    possible_moves = [field[player_x,player_y-1],field[player_x,player_y+1],field[player_x-1,player_y],field[player_x+1,player_y],field[player_x,player_y]-20]
    
    

    q_function = np.max(possible_moves)
    solution = 0
    if np.random.rand() < epsilon:
        #hacer cosas
        random_index = np.random.randint(0, len(possible_moves)-1)
        
        possible_moves[random_index] += 50
    else:
        solution=np.argmax(possible_moves)
    new_pos = player_pos
    print (ACTIONS[solution])
    if ACTIONS[solution] == 'WAIT':
        new_pos = player_pos
    elif ACTIONS[solution] == 'UP':
        new_pos = (player_x,player_y-1)
    elif ACTIONS[solution] == 'DOWN':
        new_pos = (player_x,player_y+1)
    elif ACTIONS[solution] == 'LEFT':
        new_pos = (player_x-1,player_y)
    elif ACTIONS[solution] == 'RIGHT':
        new_pos = (player_x+1,player_y)
    
    self.train_set.append((player_pos,ACTIONS[solution],new_pos))

    return ACTIONS[solution]
    
    




     
    return ACTIONS[q_function]








def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector
    return stacked_channels.reshape(-1)


#returns the closest coin to the player
def find_closest_coin_using_map(player_pos, coins, field):
    """
    Finds the closest coin to the player.
    Returns the position of the closest coin.
    """
    # Create a grid of ones and zeros
    # 1 represents a wall, 0 represents a free space
    grid = np.zeros(field.shape)
    grid[field == -100] = 0  # Wall (not passable)
    grid[field == 0] = 1  # Free space (passable)

    # Create a graph from the grid
    graph = nx.grid_2d_graph(*grid.shape)
    
    # Remove nodes that represent walls
    walls = np.argwhere(grid == 0)
    for wall in walls:
        graph.remove_node(tuple(wall))

    # Find the start node
    start = tuple(player_pos)

    # Find the closest coin
    closest_coin = None
    closest_distance = float('inf')

    for coin in coins:
        try:
            # Use A* to find the shortest path
            distance = nx.shortest_path_length(graph, start, tuple(coin), method='dijkstra')
            if distance < closest_distance:
                closest_coin = tuple(coin)
                closest_distance = distance
        except nx.NetworkXNoPath:
            # No path found to this coin
            continue

    return closest_coin

def calculate_path_to_coin(player_pos, coin_pos, field):
    """
    Calculates the path to the coin using Dijkstra's algorithm.
    Returns the path to the coin.
    """
    # Create a grid of ones and zeros
    # 1 represents a wall, 0 represents a free space
    grid = np.zeros(field.shape)
    grid[field == -100] = 0  # Wall (not passable)
    grid[field == 0] = 1  # Free space (passable)

    # Create a graph from the grid
    graph = nx.grid_2d_graph(*grid.shape)
    
    # Remove nodes that represent walls
    walls = np.argwhere(grid == 0)
    for wall in walls:
        graph.remove_node(tuple(wall))

    # Find the start node
    start = tuple(player_pos)

    # Find the path to the coin
    try:
        path = nx.shortest_path(graph, start, tuple(coin_pos), method='dijkstra')
    except nx.NetworkXNoPath:
        # No path found to this coin
        path = []

    return path