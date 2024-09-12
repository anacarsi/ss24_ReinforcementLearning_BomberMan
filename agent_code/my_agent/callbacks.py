import os
import pickle
from . import hyperparameters as hyp
import numpy as np
import networkx as nx
from settings import ROWS, COLS
from objective import Objective


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup Q-learning agent. This is called once when loading each agent.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up Q-learning agent from scratch.")
        # Initialize Q-table with zeros
        self.q_table = {}
        self.alpha = 0.9  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0
    else:
        self.logger.info("Loading Q-learning model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
        self.epsilon = 0.1
        self.epsilon_min = 0
        self.learning_rate = 0.001 
        self.discount_factor = 0.99 # To modify


def act(agent, game_state: dict) -> str:
    """
    Decide the next action based on the current game state using an epsilon-greedy policy and Q-values.
    
    :param game_state: A dictionary containing the current state of the game.
    :return: The action to be taken as a string.
    """
    state = state_to_features(game_state)
    if random.random() < self.epsilon:
        # Exploration: Choose a random action
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS)
    else:
        # Exploitation: Choose the best action based on Q-values
        self.logger.debug("Querying Q-table for action.")
        q_values = self.q_table.get(tuple(state), np.zeros(NUM_ACTIONS))
        best_action_index = np.argmax(q_values)
        return ACTIONS[best_action_index]
    
    #####################
    
    field = game_state['field']
    can_place_bomb = game_state['self'][2]
    agent_x, agent_y = game_state['self'][3]
    objective = Objective(game_state, task=1)

    # Determine the nearest objective's position (Task 1)
    agent.logger.info('Task 1: Collect coins as quickly as possible')
    nearest_objective = objective.objective(game_state, task=1)
    dist_to_objective = (nearest_objective[0] - agent_x, nearest_objective[1] - agent_y)
    
    # Get the state of adjacent tiles, handling boundary conditions
    up_state = field[agent_x, agent_y-1] if agent_y > 0 else -1
    down_state = field[agent_x, agent_y+1] if agent_y < ROWS - 1 else -1
    left_state = field[agent_x-1, agent_y] if agent_x > 0 else -1
    right_state = field[agent_x+1, agent_y] if agent_x < COLS - 1 else -1

    # Define the current state as a tuple
    self.state = (dist_to_objective, up_state, down_state, left_state, right_state, can_place_bomb)

    # Epsilon-greedy strategy to choose the next action
    if np.random.random() > self.epsilon:
        # Exploit: Choose the action with the highest Q-value for the current state
        action_values = self.q_table.get(self.state, [-10] * len(ACTIONS))
        action = np.argmax(action_values)
    else:
        # Explore: Choose a random action
        action = np.random.randint(len(ACTIONS))
    
    self.action = action 

    # Return the chosen action as a string
    return ACTIONS[self.action]

def state_to_features(game_state: dict) -> np.array:
    """
    Converts the game state to the input of the Q-learning agent.
    """
    if game_state is None:
        return None

    field = game_state['field']
    features = []
    
    # Example feature extraction:
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
