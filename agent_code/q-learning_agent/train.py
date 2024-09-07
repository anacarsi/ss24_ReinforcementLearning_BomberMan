from pickle import dump as pickle_dump
from typing import List
import numpy as np
import events as e
import networkx as nx
from json import dump as json_dump
from . import auxiliary_functions as a

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
    #OLD SITUATION 
    old_field = a.build_field(self, old_game_state)
    old_player_pos = old_game_state['self'][3]
    old_objective = a.set_objetive(self, old_game_state)
    
    #BUILD NEW FIELD AND NEW STATE
    new_field = a.build_field(self, new_game_state)
    new_can_place_bomb = new_game_state['self'][2]
    new_self_pos = new_game_state['self'][3]
    new_self_x, new_self_y = new_self_pos
    new_objective = a.set_objetive(self, new_game_state)   
    new_aprox_dist_obj = (new_objective[0] - new_self_x, new_objective[1] - new_self_y)
    new_up_state = new_field[new_self_x, new_self_y -1]
    new_down_state = new_field[new_self_x , new_self_y +1]
    new_left_state = new_field[new_self_x -1, new_self_y ]
    new_right_state = new_field[new_self_x +1, new_self_y]

    new_state = (new_aprox_dist_obj, new_up_state, new_down_state, new_left_state, new_right_state, new_can_place_bomb)

    #REWARD CALCULATION (STEP)
    reward = 0
    #reward moving effciently on the board
    # if "INVALID_ACTION" in events:
    #     reward += -1 
    # #penalize dropping bomb
    # if "BOMB_DROPPED" in events:
    #     reward += -1
    #reward for getting closer to objective
    #if there is no objective right now, we still need this variable to exist.
    new_distance_to_objective = (0,0)
    #THE AGENT DOESNT UNDERSTAND HOW WE ARE ASSIGNING WEIGHTS TO modified DIJKSTRA
    #old_distance_to_objective = a.calculate_dist_to_obj_(old_player_pos, old_objective, old_field)
    #new_distance_to_objective =a.calculate_dist_to_obj(new_self_pos, old_objective, old_field)
    #if len(new_distance_to_objective) < len(old_distance_to_objective):
    #    reward += -1
    #else:
    #     reward += -3
    #SAME WITH NORMAL DIJKSTRA
    old_distance_to_objective = a.dijkstra(old_player_pos, old_objective, old_field)
    new_distance_to_objective =a.dijkstra(new_self_pos, old_objective, old_field)
    if old_objective == new_objective:
        if len(new_distance_to_objective) < len(old_distance_to_objective):
            reward += 1
        else:
            reward += -3
    #SAME WITH APROX STRAIGHT LINE DISTANCE
    #old_distance_to_objective = self.state[0]
    #new_distance_to_objective = new_objective[0] - new_self_x, new_objective[1] - new_self_y
    #if old_objective == new_objective:
        #if the new distance is smaller than the old distance, we give a reward of 1, otherwise we give a reward of -1
    #    if np.linalg.norm(new_distance_to_objective) < np.linalg.norm(old_distance_to_objective):
    #        reward += 2
    #    else:
    #        reward += -1
    #reward getting objective
    
    # if "KILLED_OPONENT" in events:
        # reward += 100
    #if "CRATE_DESTROYED" in events:
    #    reward += 3
    #if "COIN_FOUND" in events:
    #    reward += 1
    
    #UPDATE Q-TABLE
    if "COIN_COLLECTED" in events:
        reward += 2
        new_q = 5
    else:
        max_future_q = np.max(self.q_table.get(new_state, [-10] * 6))
        old_q = self.q_table.get(self.state, [-10] * 6)[self.action]
        #q function (returning qvalue for the current state and action)
        new_q = (1 - self.learning_rate) * old_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
    self.total_reward += reward
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6
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
    if 'KILLED_SELF' in events:
        reward += -100
        if self.state not in self.q_table:
            self.q_table[self.state] = [-10] * 6
        self.q_table[self.state][self.action] = -1000
    #if ("COIN_COLLECTED" in events) and ("KILLED_SELF" not in events):
    #    reward += 5
    #elif ("COIN_COLLECTED" in events) and ("KILLED_SELF" in events):
    #    reward += -10
    # if "KILLED_OPONENT" in events:
    #     reward += 100
    #if ("CRATE_DESTROYED" in events) and ("KILLED_SELF" not in events):
        #reward += 10
    self.total_reward += reward

    #UPDATE Q-TABLE AND Q-FUNCTION
    
    
    
    

    self.graph.append((self.eps, self.total_reward))
    self.total_reward = 0
    if self.eps % 1000 == 0:
        with open("graph.txt", "w") as file:
            json_dump(self.graph, file)
        # Store the Q table
        with open("q_table.pickle", "wb") as file:
            pickle_dump(self.q_table, file)
    self.eps += 1
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    #if self.eps % 2000 == 0 and self.epsilon > 0.5 and self.eps <10000:
    #     self.epsilon = 0.9



