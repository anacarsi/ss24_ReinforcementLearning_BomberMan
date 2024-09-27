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
    if x < 0 or y >= field.shape[0] or x < 0 or y >= field.shape[1]:
        return False  # Out of bounds
    if direction == "down":
        return field[x, y + 1] != -1 and field[x, y + 1] != 1
    elif direction == "up":
        return field[x, y - 1] != -1 and field[x, y - 1] != 1
    elif direction == "left":
        return field[x - 1, y] != -1 and field[x - 1, y] != 1
    elif direction == "right":
        return field[x + 1, y] != -1 and field[x + 1, y] != 1
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

def nearest_crate(game_state, x: int, y: int) -> tuple:
    """
    Find the nearest crate to the player's position.

    :param game_state: The current game state.
    :param x: The x-coordinate of the player.
    :param y: The y-coordinate of the player.
    :return: The coordinates of the nearest crate.
    """
    field = game_state["field"]
    crates = np.argwhere(field == 1)
    if len(crates) == 0:
        return None
    nearest = min(crates, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
    
    # Validate the nearest crate coordinates
    if nearest[0] >= 0 and nearest[1] >= 0:
        return nearest
    else:
        return None 
    
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

def nearest_bomb_and_range(bombs: np.ndarray, game_state: dict, xp: int, yp: int) -> tuple:
    """
    Find the nearest bomb to the player's position and calculate the explosion range of that bomb.

    :param bombs: A 17x17 numpy array where each cell contains either 0 (no bomb) or the bomb's timer.
    :param game_state: The current game state dictionary.
    :param xp: The x-coordinate of the player.
    :param yp: The y-coordinate of the player.
    :return: The coordinates of the nearest bomb and its explosion range.
    """
    if bombs is None or not np.any(bombs):
        return (None, None), []
    
    # Find all bomb positions (where the bomb timer is > 0)
    bomb_positions = np.argwhere(bombs > 0)

    # If there are no bombs, return None
    if bomb_positions.size == 0:
        print("No bombs found")
        return None, []

    # Find the nearest bomb using Manhattan distance
    nearest_bomb = min(bomb_positions, key=lambda b: abs(b[0] - xp) + abs(b[1] - yp))
    xb, yb = nearest_bomb  # Coordinates of the nearest bomb

    # Calculate the explosion range of the nearest bomb
    explosion_range = bomb_explosion_range(xb, yb, game_state["field"])

    return (xb, yb), explosion_range


def state_to_features(game_state: dict) -> Features:
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
    if xb is None or yb is None:
        x_bomb, y_bomb = -2, -2  # No bomb found
    else:
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
    nearest_crate_pos = nearest_crate(game_state, xp, yp)
    if nearest_crate_pos is not None:
        x_crate = np.sign(nearest_crate_pos[0] - xp)
        y_crate = np.sign(nearest_crate_pos[1] - yp)
    else:
        x_crate = y_crate = 0

    safe_down = is_safe(game_state, bomb_range, direction="down")
    safe_up = is_safe(game_state, bomb_range, direction="up")
    safe_left = is_safe(game_state, bomb_range, direction="left")
    safe_right = is_safe(game_state, bomb_range, direction="right")
    safe_stay = is_safe(game_state, bomb_range, direction="stay")

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

def is_safe(game_state: dict, bomb_range: list, direction="stay") -> bool:
    """
    Determine if the player's current position or a specific direction is safe from bomb explosions.

    :param game_state: The current game state.
    :param bombs: A 2D numpy array representing bomb timers.
    :param direction: The direction to check ("stay", "up", "down", "left", "right").
    :return: True if the position is safe, False otherwise.
    """
    player_cords = game_state["self"][3]
    xp, yp = player_cords
    field = game_state["field"]
    
    # Define the new position based on the direction
    if direction == "stay":
        new_x, new_y = xp, yp
    elif direction == "up":
        new_x, new_y = xp, yp - 1
    elif direction == "down":
        new_x, new_y = xp, yp + 1
    elif direction == "left":
        new_x, new_y = xp - 1, yp
    elif direction == "right":
        new_x, new_y = xp + 1, yp
    else:
        raise ValueError("Invalid direction specified.")

    # Check if the player's current position is safe
    if (xp, yp) in bomb_range:
        return False

    # Check if the new position is safe
    if (new_x, new_y) in bomb_range:
        return False

    return True  # Both current and new positions are safe


def act(self, game_state: dict) -> str:
    """
    Choose the next action based on the current state using an epsilon-greedy policy and Q-values.

    :param game_state: The current state of the game.
    :return: The chosen action as a string.
    """
    # print(f"elements of q table: {len(self.q_table)}") 

    features = state_to_features(game_state)  # Extract features
    self.state = (
        features.x_bomb, features.y_bomb,
        features.x_coin, features.y_coin,
        features.x_crate, features.y_crate,
        features.safe_down, features.safe_up,
        features.safe_left, features.safe_right,
        features.safe_stay,
        features.possible_down, features.possible_up,
        features.possible_right, features.possible_left,
        features.can_place_bomb
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

    # Define a dictionary to easily check safety and movement
    action_checks = {
        0 : (features.safe_up, features.possible_up),
        1 : (features.safe_down, features.possible_down),
        2 : (features.safe_left, features.possible_left),
        3 : (features.safe_right, features.possible_right),
        4 : (features.safe_stay, True),  # WAIT is always possible
        5 : (features.can_place_bomb, True)  # Can place bomb if allowed
    }
    """
    print(f"Current position: {game_state['self'][3]}")
    print(f"There is a bomb at: {features.x_bomb, features.y_bomb}")
    print(f"Nearest coin in direction: {features.x_coin, features.y_coin}")
    print(f"Nearest crate in direction: {features.x_crate, features.y_crate}")
    print(f"possible_up: {features.possible_up}, safe_up: {features.safe_up}")
    print(f"possible_down: {features.possible_down}, safe_down: {features.safe_down}")
    print(f"possible_left: {features.possible_left}, safe_left: {features.safe_left}")
    print(f"possible_right: {features.possible_right}, safe_right: {features.safe_right}")
    """
    
    valid_actions = [action for action, string in enumerate(ACTIONS) if action_checks[action][0] and action_checks[action][1]]

    # If no valid actions are available, fallback to a default action
    if not valid_actions:
        action = 4  # WAIT
    else:
        # Choose a random action from valid actions
        action = np.random.choice(valid_actions)

    # Set the current action for later use
    self.action = action

    # print(f"Choosing action: {ACTIONS[action]}")
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