import os
from pickle import load as pickle_load
import numpy as np
from . import auxiliary_functions as a
from . import hyperparameters as hyp


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

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

    #HYPERPARAMETERS
    hyp.training_hyperparameters(self)

    
    self.logger.info("Loading q-table from saved state.")
    #we open it if it id not empty, otherwise we create it as empty and then random initialize it
    if os.path.isfile("q_table.pickle"):
        with open("q_table.pickle", "rb") as file:
            self.q_table = pickle_load(file)
    else:
        self.q_table = {}

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    field = a.build_field(self, game_state)
    can_place_bomb = game_state['self'][2]

    explosions = a.bomb_map(self, game_state)

    vision1 = a.agent_vision(self, field, game_state['self'][3], hyp.VISION_RANGE)
    vision2 = a.agent_vision(self, explosions, game_state['self'][3], hyp.VISION_RANGE)

    self.state = (vision1.tobytes(), vision2.tobytes(), can_place_bomb)
    

    #epsilon-greedy strategy
    if np.random.random() > self.epsilon:
        action = np.argmax(self.q_table.get(self.state, [-10] * 6))
    else:
        action = np.random.randint(0, 6)
    self.action = action 
    
    return ACTIONS[action]