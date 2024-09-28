import os
import sys
import pickle
import numpy as np
import networkx as nx
from typing import NamedTuple

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup Q-learning agent. This is called once when loading each agent.
    """
    self.q_table = {}
    self.epsilon = 0.9
    self.epsilon_decay = 0.9999
    self.epsilon_min = 0.1
    self.discount_factor = 0.8
    self.logger.info("Model initialized")
    load_model(self)
    self.logger.info("Model loaded")

def get_dir() -> str:
        return os.path.dirname(__file__)

def load_model(self):
    """
    Load the Q-table from a file if it exists.
    """
    q_table = {}
    if os.path.isfile(get_dir() + '/q_table.pickle'):
        with open(get_dir() + '/q_table.pickle', 'rb') as file:
            q_table = pickle.load(file)
            self.q_table = q_table
            self.logger.info("Model loaded")
            print("Model loaded")
    else:
        self.logger.info("Model not found")

class Features(NamedTuple):
    xdirection_nearest_bomb: int    # Actual coordinates of the nearest bomb
    ydirection_nearest_bomb: int
    xdirection_nearest_coin: int
    ydirection_nearest_coin: int
    bomb_map: np.ndarray
    bomb_range: np.ndarray
    reduced_field: np.ndarray
    can_place_bomb: bool


def new_coordinate_map(map_data: np.ndarray, size: int, xp: int, yp: int) -> tuple:

    # Calculate distances to the edges
    distance_to_top = yp
    distance_to_bottom = map_data.shape[0] - yp
    distance_to_left = xp
    distance_to_right = map_data.shape[1] - xp

    # Adjust y axis
    if distance_to_top < 3:
        yp_cropped = yp # New agent position will be centered
    elif distance_to_top >= 3 and distance_to_top <= 13:
        yp_cropped = 3
    elif distance_to_top > 13 and distance_to_top <= 16:
        yp_cropped = size - distance_to_bottom
    else: 
        raise ValueError("Invalid distance")
    
    # Adjust x axis
    if distance_to_left < 3:
        xp_cropped = xp
    elif distance_to_left >= 3 and distance_to_left <= 13:
        xp_cropped = 3
    elif distance_to_left > 13 and distance_to_left <= 16:
        xp_cropped = size - distance_to_right
    else: 
        raise ValueError("Invalid distance")
    
    return (xp_cropped, yp_cropped)

def construct_map(game_state: dict, type: str, size=7) -> np.ndarray:
    """
    Construct a 7x7 field/explosion map/bomb map centered on the player from the current game state.

    :param game_state: The current game state.
    :param type: The type of map to construct ("field", "explosion_map", "bomb_map").
    :param size: The size of the cropped map.
    :return: A 7x7 numpy array representing the cropped game field.
    """
    half_size = size // 2 - 1
    map_data = None
    
    # Handling different map types
    if type == "field":
        map_data = np.array(game_state["field"])  # Convert to numpy array
    elif type == "explosion_map":
        map_data = np.array(game_state["explosion_map"])  # Convert to numpy array
    elif type == "bomb_map":
        # Create a bomb map of the same size as the field and initialize with zeros
        map_data = np.zeros_like(game_state["field"], dtype=float)
        
        # Place bomb timers into the map
        for (xb, yb), timer in game_state["bombs"]:
            map_data[xb, yb] = timer + 1.0
    else:
        raise ValueError("Invalid map type specified.")

    player_coords = game_state["self"][3]
    xp, yp = player_coords

    xp_new, yp_new = new_coordinate_map(map_data, size=7, xp=xp, yp=yp)
    
    # Generate agent vision of size 7x7 centered on the player xp_new, yp_new
    new_map = map_data[xp_new - half_size: xp_new + half_size + 1, yp_new - half_size: yp_new + half_size + 1]

    return new_map


def is_move_possible(game_state, x: int, y: int, direction: str) -> bool:
    """
    Determine if a move is possible in the specified direction.

    :param game_state: The current game state.
    :param x: The x-coordinate of the player.
    :param y: The y-coordinate of the player.
    :param direction: The direction to check.
    :return: True if the move is possible, False otherwise.
    """
    field = game_state["field"]
    if x < 0 or y >= field.shape[0] or x < 0 or y >= field.shape[1]:
        return False  # Out of bounds
    if direction == ACTIONS[1]:  # down
        return field[x, y + 1] != -1 and field[x, y + 1] != 1
    elif direction == ACTIONS[0]:  # up
        return field[x, y - 1] != -1 and field[x, y - 1] != 1
    elif direction == ACTIONS[2]:  # left
        return field[x - 1, y] != -1 and field[x - 1, y] != 1
    elif direction == ACTIONS[3]:   # right
        return field[x + 1, y] != -1 and field[x + 1, y] != 1
    return True
    
def nearest_coin(field_cropped: np.ndarray, x: int, y: int) -> tuple:
    """
    Find the nearest coin to the player's position.

    :param game_state: The current game state.
    :param x: The x-coordinate of the player.
    :param y: The y-coordinate of the player.
    :return: The coordinates of the nearest coin.
    """
    # Find the nearest coin inside the player's field of view
    coins = np.argwhere(field_cropped == 3)
    if coins.size == 0:
        # print("No coins found")
        return (-17, -17)  # No coins found
    return min(coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y))

def nearest_bomb(bomb_map: np.ndarray, x: int, y: int) -> tuple:
    """
    Find the nearest bomb to the player's position.

    :param bomb_map: The current bomb_map 7x7 that the agent sees.
    :param x: The x-coordinate of the player.
    :param y: The y-coordinate of the player.
    :return: The coordinates of the nearest bomb.
    """
    bomb_positions = np.argwhere(bomb_map > 0)
    if bomb_positions.size == 0:
        # print("No bombs found")
        return (-17, -17)  # No bombs found
    return min(bomb_positions, key=lambda b: abs(b[0] - x) + abs(b[1] - y))
    
def bomb_explosion_range(xb: int, yb: int, field: np.array) -> list:
    """
    Calculate the tiles affected by the explosion of a bomb at (xb, yb).
    The explosion extends up to 3 tiles in each direction unless blocked by a stone wall.

    :param xb: x-coordinate of the bomb.
    :param yb: y-coordinate of the bomb.
    :param field: The game field, with walls and empty spaces.
    :return: A list of coordinates affected by the bomb explosion.
    """
    explosion_range = [(xb, yb)]  # The bomb's own position
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

    for dx, dy in directions:
        for step in range(1, 4):  # Bomb affects up to 3 tiles in each direction
            nx, ny = xb + dx * step, yb + dy * step
            if field[nx, ny] == -1:  # Stop at stone walls
                break
            explosion_range.append((nx, ny))

    return explosion_range

def state_to_features(game_state: dict) -> Features:
    """
    Define the state of the game as a feature vector.

    :param game_state: The current game state.
    :return: The feature vector representing the game state.
    """
    player_coords = game_state["self"][3]  # Player's coordinates
    xp, yp = player_coords

    # Nearest bomb
    bomb_map = construct_map(game_state, "bomb_map")
    (xb, yb) = nearest_bomb(bomb_map, xp, yp)
    # In which direction the nearest bomb is
    xdirection_nearest_bomb = xb
    ydirection_nearest_bomb = yb
    print(f"The bomb map is: {bomb_map}")

    # Nearest coin
    field_cropped = construct_map(game_state, "field")
    (xc, yc) = nearest_coin(field_cropped, xp, yp)
    xdirection_nearest_coin = xc
    ydirection_nearest_coin = yc
    print(f"The field is: {field_cropped}")

    """
    print("My nearest bomb is at: ", xb, yb)
    print("The direction of the nearest bomb is: ", xdirection_nearest_bomb, ydirection_nearest_bomb)
    print("My position is: ", xp, yp)
    print("The bomb map is: ", bomb_map)
    print("The field is: ", construct_map(game_state, "field"))
    """

    # Extracting the explosion map and cropping to 7x7
    explosion_map = construct_map(game_state, "explosion_map")
    print(f"The explosion map is: {explosion_map}")
    can_place_bomb = game_state["self"][2]

    # Return the feature vector
    return Features(
        xdirection_nearest_bomb=xdirection_nearest_bomb, ydirection_nearest_bomb=ydirection_nearest_bomb,
        xdirection_nearest_coin=xdirection_nearest_coin, ydirection_nearest_coin=ydirection_nearest_coin,
        bomb_map=bomb_map, reduced_field=construct_map(game_state, "field"),
        bomb_range=explosion_map,
        can_place_bomb=can_place_bomb
    )

def is_move_safe(game_state: dict, bomb_range: np.ndarray, direction: str) -> bool:
    """
    Determine if the player's current position or a specific direction is safe from bomb explosions.

    :param game_state: The current game state.
    :param bombs: A 2D numpy array representing bomb timers.
    :param direction: The direction to check ("stay", "up", "down", "left", "right").
    :return: True if the position is safe, False otherwise.
    """
    xp, yp = game_state["self"][3]
    
    # Define the new position based on the direction
    if direction == ACTIONS[4] or direction == ACTIONS[5]:  # STAY / BOMB
        new_x, new_y = xp, yp
    elif direction == ACTIONS[0]: # UP
        new_x, new_y = xp, yp - 1
    elif direction == ACTIONS[1]: # DOWN   
        new_x, new_y = xp, yp + 1
    elif direction == ACTIONS[2]: # LEFT
        new_x, new_y = xp - 1, yp
    elif direction == ACTIONS[3]: # RIGHT
        new_x, new_y = xp + 1, yp
    else:
        raise ValueError("Invalid direction specified.")
    
    # Check if the new position is safe
    """
    print(f"Current position: ({xp}, {yp})")
    print(f"Direction: {direction}")
    print(f"New position: ({new_x}, {new_y})")"""
    if (new_x, new_y) in bomb_range and direction != ACTIONS[4] and direction != ACTIONS[5]:
        print(" IF I MOVE I WILL BE IN THE BOMB RANGE")
        return False    
    if (new_x, new_y) in bomb_range and direction == ACTIONS[4] and direction != ACTIONS[5]:
        print(" IF I STAY I WILL BE IN THE BOMB RANGE")
        return False
    
    return True  # Both current and new positions are safe


def convert_to_hashable(item):
    """
    Converts unhashable types (like numpy arrays, lists, dicts) to hashable ones.
    - Converts NumPy arrays to bytes using .tobytes()
    - Recursively handles lists and dicts
    """
    if isinstance(item, np.ndarray):
        return item.tobytes()  # Convert NumPy array to a byte representation
    elif isinstance(item, list):
        return tuple(convert_to_hashable(x) for x in item)  # Convert lists to tuples
    elif isinstance(item, dict):
        return tuple((key, convert_to_hashable(value)) for key, value in item.items())  # Convert dicts to tuples
    else:
        return item  # If the item is already hashable, return as-is

def act(self, game_state: dict) -> str:
    """
    Choose the next action based on the current state using an epsilon-greedy policy and Q-values.

    :param game_state: The current state of the game.
    :return: The chosen action as a string.
    """
    features = state_to_features(game_state)  # Extract features
    self.state = (
        convert_to_hashable(features.xdirection_nearest_bomb),  # Convert individual features to hashable
        convert_to_hashable(features.ydirection_nearest_bomb),
        convert_to_hashable(features.xdirection_nearest_coin),
        convert_to_hashable(features.ydirection_nearest_coin),
        convert_to_hashable(features.bomb_map),  # Convert bomb_map to hashable type
        convert_to_hashable(features.bomb_range),
        convert_to_hashable(features.reduced_field),  # Convert reduced_field to hashable type
        convert_to_hashable(features.can_place_bomb)
    )
    # Hashable state for Q-table lookup
    hashed_state = tuple(self.state)

    # Epsilon-greedy policy for action selection
    if np.random.random() > self.epsilon:
        if hashed_state in self.q_table:
            action = np.argmax(self.q_table[hashed_state])
        else:
            action = np.random.randint(0, len(ACTIONS))  # Explore
    else:
        action = np.random.randint(0, len(ACTIONS))  # Explore

    # Set the current action for later use
    # print(f"Choosing action: {ACTIONS[action]}")
    self.action = action
    self.logger.info(f"Choosing action: {ACTIONS[action]}")

    # Return the action corresponding to the chosen index
    return ACTIONS[action]

"""
def apply_mutations_to_action(x_flip, y_flip, transpose, action):
    
    x_flip_map = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    y_flip_map = {"UP": "DOWN", "DOWN": "UP"}
    transpose_map = {"UP": "LEFT", "LEFT": "UP", "DOWN": "RIGHT", "RIGHT": "DOWN"}

    if transpose and action in transpose_map:
        action = transpose_map[action]
    if x_flip and action in x_flip_map:
        action = x_flip_map[action]
    if y_flip and action in y_flip_map:
        action = y_flip_map[action]

    return action

"""