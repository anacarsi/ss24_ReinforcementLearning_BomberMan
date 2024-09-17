import os
import sys
import pickle
import numpy as np
import networkx as nx
from objective import Objective

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import settings as s

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

def act(self, game_state: dict) -> str:
    """
    Decide the next action based on the current game state using an epsilon-greedy policy and Q-values.
    
    :param game_state: A dictionary containing the current state of the game.
    :return: The action to be taken as a string.
    """
    action = self.model.choose_action(game_state)

    self.logger.info(f"Choosing action: {action} for state: {game_state}")

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of the Q-learning agent.
    """
    if game_state is None:
        return None

    field = game_state['field']
    features = []
    
    # 1. Player position
    player_pos = game_state['self'][3]
    features.append(player_pos[0])
    features.append(player_pos[1])
    
    # 2. Coins and Crates
    coins = game_state['coins']
    crates = np.argwhere(field == -2)
    features.append(len(coins))
    features.append(len(crates))

    # Convert features to a numpy array
    return np.array(features)

    """
    import numpy as np

def state_to_features(self, game_state: dict) -> np.ndarray:
    
    Converts the game state to the input of the Q-learning agent, including field and bomb map calculations.

    :param game_state: Dictionary describing the current game state.
    :return: Numpy array of features including the field and bomb map.
    
    if game_state is None:
        return None

    field = game_state['field']
    bombs = game_state.get('bombs', [])
    explosion_map = game_state.get('explosion_map', np.zeros((7, 7), dtype=int))
    
    # Initialize bomb map
    bomb_map = np.full((7, 7), 6)

    # Process bombs to update the bomb map
    for bomb in bombs:
        bomb_center_x, bomb_center_y = bomb[0]
        bomb_timer = bomb[1] + 1

        # Define affected tiles for each direction
        affected_tiles_up = [(bomb_center_x, bomb_center_y), (bomb_center_x, bomb_center_y - 1)]
        affected_tiles_down = [(bomb_center_x, bomb_center_y + 1)]
        affected_tiles_left = [(bomb_center_x - 1, bomb_center_y)]
        affected_tiles_right = [(bomb_center_x + 1, bomb_center_y)]

        # Update bomb map for each direction
        for tile in affected_tiles_up + affected_tiles_down + affected_tiles_left + affected_tiles_right:
            x, y = tile
            if 0 <= x < 7 and 0 <= y < 7 and field[x, y] != -1:
                bomb_map[x, y] = min(bomb_map[x, y], bomb_timer)

    # Adjust field based on crates, coins, explosions, and bomb map
    field_adjusted = field.copy()
    field_adjusted[field == 1] = -2  # Crates

    # Set explosion tiles to 1
    field_adjusted[explosion_map == 1] = 1

    # Update field based on bomb map countdown
    for i in range(1, 5):
        field_adjusted[bomb_map == i] = i

    # Update field with coins
    coins = game_state.get('coins', [])
    objective = self.set_objetive(game_state)
    if objective in coins and field_adjusted[objective] not in {0, 1, 2, 3, 4}:
        field_adjusted[objective] = -3

    # Extract features
    player_pos = game_state['self'][3]
    features = [
        player_pos[0],  # Player x position
        player_pos[1],  # Player y position
        len(coins),     # Number of coins
        len(np.argwhere(field == -2))  # Number of crates
    ]

    # Combine features and field into a single array
    features = np.array(features)
    return features, field_adjusted, bomb_map

    """

    """
    import numpy as np

def generate_view_explosion(self, game_state: dict) -> np.ndarray:
    
    Generates a bomb map based on the given game state. Each tile in the bomb map contains the timer value of the nearest bomb.
    Parameters:
    - game_state (dict): The current game state containing the field and bomb information.
    Returns:
    - bomb_map: A 2D array representing the bomb map, where each element represents the timer value of a bomb.
    
    field = game_state['field']
    bombs = game_state.get('bombs', [])
    
    # Initialize bomb map with a large timer value (6, which is more than the maximum bomb timer)
    bomb_map = np.full((7, 7), 6)

    # Process each bomb
    for (bomb_pos, bomb_timer) in bombs:
        bomb_x, bomb_y = bomb_pos
        bomb_timer += 1  # Increment bomb timer to reflect current state
        
        # Directions: up, down, left, right
        directions = [
            (0, -1),  # Up
            (0, 1),   # Down
            (-1, 0),  # Left
            (1, 0)    # Right
        ]
        
        # Update bomb map for the bomb's position and its affected tiles
        for dx, dy in directions:
            x, y = bomb_x, bomb_y
            while True:
                x += dx
                y += dy
                
                # Check bounds and walls
                if not (0 <= x < 7 and 0 <= y < 7) or field[x, y] == -1:
                    break
                
                # Update bomb map with the minimum timer value
                bomb_map[x, y] = min(bomb_map[x, y], bomb_timer)
    
    return bomb_map

    """

class QLearningModel():
    def __init__(self):
        self.q_table = {}
        self.epsilon = 1.0  # Exploration rate for epsilon-greedy
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.q_table = {} 
        self.task = 2
        self.define_task(self.task)

    def define_task(self, task) -> None:
        """
        Define actions depending on the task.
        Task 1: Collect coins as quickly as possible.
        Task 2: Collect hidden coins within a step limit.
        Task 3: Hunt and blow up a predefined peaceful agent.
            Task 3.2: Hunt and blow up a coin collector agent.
        Task 4: Survive and collect coins.

        :param task: The task to be performed.
        """
        if task == 1:
            self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
        elif task > 1:
            self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    
    def load_table(self, q_table) -> None:
        """
        Load the Q-table from a file if it exists.

        :param q_table: The Q-table to be loaded.
        """
        self.q_table = q_table
    
    def choose_action(self, state) -> str:
        """
        Choose the next action based on the current state using an epsilon-greedy policy and Q-values.

        :param state: The current state of the game.
        """
        objective = Objective(self.task, state)
        closest_objective = objective.set_objective()

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            candidates = self.q_table.get(tuple(state), np.zeros(len(self.actions)))
            action = self.actions[np.argmax(candidates)]
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action