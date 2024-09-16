# callbacks.py
import os
import sys
import pickle
import numpy as np
import networkx as nx

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import settings as s

def setup(self):
    """
    Setup Q-learning agent. This is called once when loading each agent.
    """
    self.model = QLearningModel()
    self.logger.info("Model initialized")
    self.load_model()

def load_model(self):
    """
    Load the Q-table from a file if it exists.
    """
    q_table = {}
    if os.path.isfile(get_dir() + '/q_table.pkl'):
        with open(get_dir() + '/q_table.pkl', 'rb') as file:
            q_table = pickle.load(file)
            self.model.load_q_table(q_table)
            self.logger.info("Model loaded")
            print("Model loaded")
    else:
        self.logger.info("Model not found")

    def get_dir():
        return os.path.dirname(__file__)


def act(self, game_state: dict) -> str:
    """
    Decide the next action based on the current game state using an epsilon-greedy policy and Q-values.
    
    :param game_state: A dictionary containing the current state of the game.
    :return: The action to be taken as a string.
    """
    action = self.model.choose_action(game_state)

    self.logger.info(f"Choosing action: {action} for state: {game_state}")

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

class QLearningModel():
    def __init__(self):
        self.q_table = {}
        self.epsilon = 1.0  # Exploration rate for epsilon-greedy
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.q_table = {} 
        self.task = 1
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
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            candidates = self.q_table.get(tuple(state), np.zeros(len(self.actions)))
            action = self.actions[np.argmax(candidates)]
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action
    

