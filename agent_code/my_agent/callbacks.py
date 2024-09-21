import os
import sys
import pickle
import numpy as np
import networkx as nx

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

def act(self, game_state: dict) -> str:
    """
    Choose the next action based on the current state using an epsilon-greedy policy and Q-values.

    :param state: The current state of the game.
    """
    field, bombs, vision = state_to_features(self, game_state)
    can_place_bomb = game_state['self'][2]
    player_pos = game_state['self'][3]
    vision_bombs = create_vision(self, bombs, player_pos)
    self.state = (vision.tobytes(), vision_bombs.tobytes(), can_place_bomb)

    if np.random.random() > self.model.epsilon:
        action = np.argmax(self.q_table.get(self.state, [-10] * 6))
    else:
        action = np.random.randint(0, 6)

    self.action = action 
    self.logger.info(f"Choosing action: {action} for state: {game_state}")

    return ACTIONS[action]


def state_to_features(self, game_state: dict) -> np.ndarray:
    """
    Converts the game state to the input of the Q-learning agent, including field and bomb map calculations.

    :param game_state: Dictionary describing the current game state.
    :return: Numpy array of features including the field, bomb map, and agent's vision.
    """

    # Step 1: Build the game field and encode relevant entities
    field = field_to_features(self, game_state)
    
    # Step 2: Generate the bomb map based on bomb timers and positions
    bomb_map = bombs_to_features(self, game_state, field)
    
    # Step 3: Extract the agent's vision with a specific radius
    player_pos = game_state['self'][3]
    vision = create_vision(self, field, player_pos)
    
    # Step 4: Flatten and concatenate field, bomb map, and vision into a single feature vector
    return field, bomb_map, vision

def bombs_to_features(self, game_state: dict, field: np.ndarray) -> np.ndarray:
    """
    Generates a bomb map based on the game state, considering bomb timers and walls.
    
    :param game_state: The dictionary containing the current game state.
    :param field: The game field with walls and obstacles.
    :return: A bomb map as a 2D numpy array, indicating bomb timers.
    """
    bomb_map = np.full_like(field, -1)  # Initialize bomb map with -1 (no bomb)
    
    for (bomb_pos, bomb_timer) in game_state['bombs']:
        x, y = bomb_pos
        timer = bomb_timer + 1  # Timer counts down

        # Define affected tiles (current bomb and its potential explosion range)
        affected_tiles = [
            [(x, y), (x, y - 1)],  # Up
            [(x, y + 1)],          # Down
            [(x - 1, y)],          # Left
            [(x + 1, y)]           # Right
        ]

        # Update bomb map for all affected tiles unless blocked by walls (-1 in field)
        for direction in affected_tiles:
            for tile in direction:
                if not (0 <= tile[0] < field.shape[0] and 0 <= tile[1] < field.shape[1]):
                    break  # Ensure tile is within bounds
                if field[tile] == -1:
                    break  # Wall blocks the bomb's effect
                bomb_map[tile] = timer

    # Incorporate any active explosions from the game state (explosions set the bomb timer to 1)
    bomb_map[game_state['explosion_map'] == 1] = 1

    return bomb_map

def field_to_features(self, game_state: dict) -> np.ndarray:
    """
    Builds the game field with adjusted values for bombs, coins, and opponents.
    
    :param game_state: The dictionary containing the current game state.
    :return: The game field encoded with relevant entities.
    """
    field = game_state['field'].copy()  # Copy the field (walls and empty spaces)
    
    # Encode bombs as 2 on the field
    for bomb_pos, _ in game_state['bombs']:
        field[bomb_pos] = 2

    # Encode coins as 3 on the field
    for coin_pos in game_state['coins']:
        field[coin_pos] = 3

    # Encode opponents as 4 on the field
    for opponent in game_state['others']:
        field[opponent[3]] = 4
    
    return field

def create_vision(self, field: np.ndarray, player_pos: tuple, radius: int = 8) -> np.ndarray:
    """
    Generates the agent's vision grid centered on the player's position.
    
    :param field: The game field.
    :param player_pos: The position of the player.
    :param radius: The radius of the vision grid around the player.
    :return: A (2*radius+1)x(2*radius+1) grid representing the agent's vision.
    """
    # Create an empty vision array
    vision = np.zeros((2 * radius + 1, 2 * radius + 1)) 
    x, y = player_pos
    # Define the area around the player within the radius
    left, right = max(0, x - radius), min(field.shape[0], x + radius + 1)
    up, down = max(0, y - radius), min(field.shape[1], y + radius + 1)
    
    # Fill the vision grid with the corresponding tiles from the field
    vision[(left - (x - radius)):(right - (x - radius)),
           (up - (y - radius)):(down - (y - radius))] = field[left:right, up:down]

    return vision


class QLearningModel():
    def __init__(self):
        self.q_table = {}
        self.epsilon = 0.9
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1
        self.discount_factor = 0.8

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
    