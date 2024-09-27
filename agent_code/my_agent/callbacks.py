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
    self.model = QLearningModel()
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
    if os.path.isfile(get_dir() + '/q_table.pkl'):
        with open(get_dir() + '/q_table.pkl', 'rb') as file:
            q_table = pickle.load(file)
            self.model.load_table(q_table)
            self.logger.info("Model loaded")
            print("Model loaded")
    else:
        self.logger.info("Model not found")

class Features(NamedTuple):
    x_bomb: int
    y_bomb: int
    x_coin: int
    y_coin: int
    x_crate: int
    y_crate: int
    safe_down: bool         # checks if moving down would place the player in a safe position
    safe_up: bool
    safe_left: bool
    safe_right: bool
    safe_stay: bool
    possible_up: bool
    possible_down: bool     # whether the move is mechanically possible
    possible_right: bool
    possible_left: bool
    can_place_bomb: bool
    bomb_range: list 
    # danger_bomb = (1 / distance_to_bomb) * (1 / bomb_timer)
    # value_coin = (1 / distance_to_coin) * (safety_modifier)


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
    if direction == "down":
        return field[x, y + 1] == 0
    elif direction == "up":
        return field[x, y - 1] == 0
    elif direction == "left":
        return field[x - 1, y] == 0
    elif direction == "right":
        return field[x + 1, y] == 0
    return True
    
def nearest_coin(game_state, x: int, y: int) -> tuple:
    """
    Find the nearest coin to the player's position.

    :param game_state: The current game state.
    :param x: The x-coordinate of the player.
    :param y: The y-coordinate of the player.
    :return: The coordinates of the nearest coin.
    """
    coins = game_state["coins"]
    if not coins:
        return None
    return min(coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y))

def nearest_bomb(game_state, x: int, y: int) -> tuple:
    """
    Find the nearest bomb to the player's position.

    :param game_state: The current game state.
    :param x: The x-coordinate of the player.
    :param y: The y-coordinate of the player.
    :return: The coordinates of the nearest bomb.
    """
    bombs = game_state["bombs"]
    if not bombs:
        return None
    return min(bombs, key=lambda b: abs(b[0][0] - x) + abs(b[0][1] - y))

def nearest_crater(game_state, x: int, y: int) -> tuple:
    """
    Find the nearest crate to the player's position.

    :param game_state: The current game state.
    :param x: The x-coordinate of the player.
    :param y: The y-coordinate of the player.
    :return: The coordinates of the nearest crate.
    """
    crates = game_state["crates"]
    if not crates:
        return None
    return min(crates, key=lambda c: abs(c[0] - x) + abs(c[1] - y))   
    
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

def nearest_bomb_and_range(bombs: np.ndarray, game_state, xp: int, yp: int) -> tuple:
    """
    Find the nearest bomb to the player's position and calculate the explosion range of that bomb.

    :param game_state: The current game state.
    :param xp: The x-coordinate of the player.
    :param yp: The y-coordinate of the player.
    :return: The coordinates of the nearest bomb and its explosion range.
    """
    if bombs is None:
        return None, []

    nearest_bomb = min(bombs, key=lambda b: abs(b[0][0] - xp) + abs(b[0][1] - yp))
    xb, yb = nearest_bomb[0]  # Coordinates of the nearest bomb
    explosion_range = bomb_explosion_range(xb, yb, game_state["field"])
    return (xb, yb), explosion_range

def state_to_features(game_state) -> Features:
    """
    Define the state of the game as a feature vector.

    :param game_state: The current game state.
    :return: The feature vector representing the game state.
    """
    xp, yp = game_state["self"][3]  # Player's coordinates

    # Nearest bomb and its explosion range
    bombs = np.zeros((17, 17), dtype=np.float32)
    for xy, bomb_timer in game_state["bombs"]:
        bombs[xy] = bomb_timer + 1.0 # Agent gets warning about newly placed bombs, avoiding confusion between a bomb about to explode (timer = 0) and an empty space.

    (xb, yb), bomb_range = nearest_bomb_and_range(bombs, game_state, xp, yp)
    x_bomb = np.sign(xb - xp)
    y_bomb = np.sign(yb - yp)

    # Nearest coin
    nearest_coin_pos = nearest_coin(game_state, xp, yp)
    if nearest_coin_pos:
        x_coin = np.sign(nearest_coin_pos[0] - xp)
        y_coin = np.sign(nearest_coin_pos[1] - yp)
    else:
        x_coin = y_coin = 0

    # Nearest crate
    nearest_crater_pos = nearest_crater(game_state, xp, yp)
    if nearest_crater_pos:
        x_crate = np.sign(nearest_crater_pos[0] - xp)
        y_crate = np.sign(nearest_crater_pos[1] - yp)
    else:
        x_crate = y_crate = 0

    safe_down = is_safe(game_state, direction="down")
    safe_up = is_safe(game_state, direction="up")
    safe_left = is_safe(game_state, direction="left")
    safe_right = is_safe(game_state, direction="right")
    safe_stay = is_safe(game_state, direction="stay")

    # Possible movement directions
    possible_down = is_move_possible(game_state, xp, yp, "down")
    possible_up = is_move_possible(game_state, xp, yp, "up")
    possible_right = is_move_possible(game_state, xp, yp, "right")
    possible_left = is_move_possible(game_state, xp, yp, "left")

    can_place_bomb = game_state["self"][2]

    # Return the feature vector
    return Features(
        x_bomb=x_bomb, y_bomb=y_bomb,
        x_coin=x_coin, y_coin=y_coin,
        x_crate=x_crate, y_crate=y_crate,
        safe_down=safe_down, safe_up=safe_up,
        safe_left=safe_left, safe_right=safe_right,
        safe_stay=safe_stay, 
        possible_up=possible_up, possible_down=possible_down, 
        possible_right=possible_right, possible_left=possible_left,
        can_place_bomb=can_place_bomb,
        bomb_range=bomb_range  # Tiles affected by the nearest bomb
    )

def is_safe(game_state, bombs: np.nadarray, direction="stay") -> bool:
    """
    Determine if the player's current position or a specific direction is safe from bomb explosions.

    :param game_state: The current game state.
    :param direction: The direction to check ("stay", "up", "down", "left", "right").
    :return: True if the position is safe, False otherwise.
    """
    player_cords = game_state["self"][3]
    bombs_ticking = bombs
    field = game_state["field"]
    xp, yp = player_cords

    for xy in bombs_ticking:
        xb, yb = xy
        explosion_range = bomb_explosion_range(xb, yb, field)
        if direction == "stay" and (xp, yp) in explosion_range:
            return False
        if direction == "down" and (xp, yp + 1) in explosion_range:
            return False
        if direction == "up" and (xp, yp - 1) in explosion_range:
            return False
        if direction == "left" and (xp - 1, yp) in explosion_range:
            return False
        if direction == "right" and (xp + 1, yp) in explosion_range:
            return False

    return True

def act(self, game_state: dict) -> str:
    """
    Choose the next action based on the current state using an epsilon-greedy policy and Q-values.

    :param game_state: The current state of the game.
    :return: The chosen action as a string.
    """
    # Extract features from the current game state
    features = state_to_features(game_state)
    
    # Construct the state for Q-table lookup based on the extracted features
    self.state = (
        features.x_bomb, features.y_bomb, 
        features.x_coin, features.y_coin, 
        features.x_crate, features.y_crate, 
        features.safe_down, features.safe_up, 
        features.safe_left, features.safe_right, 
        features.safe_stay, features.possible_down, 
        features.possible_up, features.possible_right, 
        features.possible_left, features.can_place_bomb
    )

    # Convert the state tuple to a hashable form (can also use a simpler way if needed)
    hashed_state = tuple(self.state)

    # Epsilon-greedy policy for action selection
    if np.random.random() > self.model.epsilon:
        if hashed_state in self.q_table:
            action = np.argmax(self.q_table[hashed_state])
        else:
            # If the state is unseen, return a default action or random action
            action = np.random.randint(0, len(ACTIONS))
    else:
        action = np.random.randint(0, len(ACTIONS))

    # Set the current action for later use
    self.action = action
    
    self.logger.info(f"Choosing action: {ACTIONS[action]} for state: {self.state}")

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