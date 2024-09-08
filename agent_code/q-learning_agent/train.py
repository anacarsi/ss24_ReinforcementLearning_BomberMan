from pickle import dump as pickle_dump
from typing import List
import numpy as np
import events as e
import networkx as nx
from json import dump as json_dump
from . import auxiliary_functions as a
from . import hyperparameters as hyp
def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.total_reward = 0
    self.eps = 0
    self.graph = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    
    #BUILD NEW FIELD AND NEW STATE
    new_field = a.build_field(self, new_game_state)
    new_can_place_bomb = new_game_state['self'][2]
    new_expl_map = a.bomb_map(self, new_game_state)
    new_player_pos_x, new_player_pos_y = new_game_state['self'][3]

    new_vision1 = a.agent_vision(self, new_field, (new_player_pos_x, new_player_pos_y), hyp.VISION_RANGE)
    new_vision2 = a.agent_vision(self, new_expl_map, (new_player_pos_x, new_player_pos_y), hyp.VISION_RANGE)

    new_state = (new_vision1.tobytes(), new_vision2.tobytes(), new_can_place_bomb)

    #REWARD CALCULATION (STEP)
    reward = -1
    if "CRATE_DESTROYED" in events:
       reward += 3
    if "COIN_FOUND" in events:
        reward += 1
    if "COIN_COLLECTED" in events:
        reward += 10
        
    #UPDATE Q-TABLE
    max_future_q = np.max(self.q_table.get(new_state, [-10] * 6))
    old_q = self.q_table.get(self.state, [-10] * 6)[self.action]
    #q function (returning qvalue for the current state and action)
    new_q = (1 - self.learning_rate) * old_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
    self.total_reward += reward
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6
        # print("State not in q_table training")
    self.q_table[self.state][self.action] = new_q


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #REWARD CALCULATION (END OF ROUND)
    reward = 0
    if e.SURVIVED_ROUND in events:
        reward += 100    
    if e.COIN_COLLECTED in events:
        reward += 10
    if e.KILLED_SELF in events:
        reward += -100
    if e.GOT_KILLED in events:
        reward += -100
    
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6
    self.q_table[self.state][self.action] += self.learning_rate * reward

    self.total_reward += reward
    
    #UPDATE Q-TABLE AND Q-FUNCTION
    self.graph.append((self.eps, self.total_reward))
    self.total_reward = 0
    if self.eps % 100000 == 0:
        with open("graph.txt", "w") as file:
            json_dump(self.graph, file)
        # Store the Q table
        with open("q_table.pickle", "wb") as file:
            pickle_dump(self.q_table, file)
    if self.eps % 100 == 0:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    self.eps += 1



