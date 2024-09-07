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
    #El objetivo de esta funcion es actualizar el q_table con la nueva informacion (ver lo que ha cambiado en el state)
    
    
    #REBUILD OLD FIELD
    old_field = old_game_state['field']
    old_bombmap = a.bomb_map(self, old_game_state)
    old_explosion_map = old_game_state['explosion_map']
    #build field with: [5,4,3,2] cointdown to explosion, 1 explosion, 0 free space, -1 wall, -2 crate
    # adjust scale of crates to -2
    old_field[old_field==1] = -2
    # explosions are present two steps. Where explosions are present, adjust field to show 1 (explosion currently on that tile).
    old_field[old_explosion_map==1] = 1
    old_field[old_explosion_map==2] = 1
    # adjust field to show countdown to explosions if there are currently no explosions.
    cond = ((old_bombmap == 5) | (old_bombmap == 4) | (old_bombmap == 3) | (old_bombmap == 2)) & (old_field != 1)
    old_field[cond] = old_bombmap[cond]
    # old player position
    old_player_pos = old_game_state['self'][3]
    old_objective = a.set_objetive(self, old_game_state)
    
    #REBUILD NEW FIELD
    new_field = new_game_state['field']
    new_explosion_map = new_game_state['explosion_map']
    new_bombmap = a.bomb_map(self, new_game_state)
    new_can_place_bomb = new_game_state['self'][2]
    #build new field with: [5,4,3,2] cointdown to explosion, 1 explosion, 0 free space, -1 wall, -2 crate
    # adjust scale of crates to -2
    new_field[new_field==1] = -2
    # explosions are present two steps. Where explosions are present, adjust field to show 1 (explosion currently on that tile).
    new_field[new_explosion_map == 1] = 1
    new_field [new_explosion_map == 2] = 1
    # adjust field to show countdown to explosions if there are currently no explosions.
    cond = ((new_bombmap == 5) | (new_bombmap == 4) | (new_bombmap == 3) | (new_bombmap == 2)) & (new_field != 1)
    new_field[cond] = new_bombmap[cond]
    # new position of the agent.
    new_self = new_game_state['self']
    new_self_pos = new_self[3]
    new_self_x, new_self_y = new_self_pos
    # new objective
    new_objective = a.set_objetive(self, new_game_state)   


    #if there is no objective right now, we still need this variable to exist.
    new_distance_to_objective = (0,0)
    reward = 0
    
    old_distance_to_objective = a.calculate_dist_to_obj(old_player_pos, old_objective, old_field)
    new_distance_to_objective =a.calculate_dist_to_obj(new_self_pos, old_objective, old_field)
    # if len(new_distance_to_objective) < len(old_distance_to_objective):
    #     reward += 1
    # else:
    #     reward += -2
    
    # if "KILLED_OPONENT" in events:
        # reward += 100
    # if "BOMB_DROPPED" in events:
    #     reward += -1
    if "CRATE_DESTROYED" in events:
        reward += 3
        print("CRATE_DESTROYED")
    if "COIN_FOUND" in events:
        reward += 1
        print("COIN_FOUND")
    if "COIN_COLLECTED" in events:
        reward += 5
        print("COIN_COLLECTED")
    if "INVALID_ACTION" in events:
        reward += -1
        print("INVALID_ACTION")
    
    new_up_state = new_field[new_self_x, new_self_y -1]
    new_down_state = new_field[new_self_x , new_self_y +1]
    new_left_state = new_field[new_self_x -1, new_self_y ]
    new_right_state = new_field[new_self_x +1, new_self_y]

    new_aprox_dist_obj = new_objective[0] - new_self_x, new_objective[1] - new_self_y
    new_state = (new_aprox_dist_obj, new_up_state, new_down_state, new_left_state, new_right_state, new_can_place_bomb)
    max_future_q = np.max(self.q_table.get(new_state, [-10] * 6))
    old_q = self.q_table.get(self.state, [-10] * 6)[self.action]
    
    # La q-function me devuelve un número, el cual es el q value para la acción que estoy tomando
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
    reward = 0
    if 'KILLED_SELF' in events:
        reward += -10
        print("KILLED_SELF")
    if ("COIN_COLLECTED" in events) and ("KILLED_SELF" not in events):
        reward += 10
        print("COIN_COLLECTED 1")
    elif ("COIN_COLLECTED" in events) and ("KILLED_SELF" in events):
        reward += -10
        print("COIN_COLLECTED 2")
    # if "KILLED_OPONENT" in events:
    #     reward += 100
    if ("CRATE_DESTROYED" in events) and ("KILLED_SELF" not in events):
        reward += 10
        print("CRATE_DESTROYED 1")
    self.total_reward += reward

    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6
    max_future_q = np.max(self.q_table.get(self.state, [-10] * 6))
    old_q = self.q_table.get(self.state, [-10] * 6)[self.action]
    self.q_table[self.state][self.action] = (1 - self.learning_rate) * old_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
    self.q_table.setdefault(self.state, [-10] * 6)[self.action] +=  (1 - self.learning_rate) * old_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
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
    # if self.eps % 5000 == 0 and self.epsilon > 0.01:
    #     self.epsilon = 0.9



