import numpy as np
import networkx as nx
import events as e
from json import dump as json_dump
from pickle import dump as pickle_dump
from typing import List, Dict, Tuple
from . import objective as a

def setup_training(self):
    self.total_reward = 0
    self.eps = 0
    self.graph = []

def game_events_occurred(
        self,
        old_game_state: Dict,
        self_action: str,
        new_game_state: Dict,
        events: List[str]
    ):
    """
    Handle game events and update the Q-table accordingly. Called once per step to allow intermediate rewards based on game events.

    :param old_game_state: The state before the action.
    :param self_action: The action taken.
    :param new_game_state: The state after the action.
    :param events: List of events occurred.
    """
    old_field = a.build_field(self, old_game_state)
    old_player_pos = old_game_state['self'][3]
    old_objective = a.set_objetive(self, old_game_state)

    new_field = a.build_field(self, new_game_state)
    new_self_pos = new_game_state['self'][3]
    new_objective = a.set_objetive(self, new_game_state)
    new_aprox_dist_obj = np.subtract(new_objective, new_self_pos)
    new_state = (
        new_aprox_dist_obj,
        new_field[new_self_pos[0], new_self_pos[1] - 1],
        new_field[new_self_pos[0], new_self_pos[1] + 1],
        new_field[new_self_pos[0] - 1, new_self_pos[1]],
        new_field[new_self_pos[0] + 1, new_self_pos[1]],
        new_game_state['self'][2]
    )

    reward = self.calculate_reward(
        old_player_pos, old_objective, new_self_pos, new_objective, old_field, new_field, events
    )

    max_future_q = np.max(self.q_table.get(new_state, [-10] * 6))
    old_q = self.q_table.get(self.state, [-10] * 6)[self.action]
    new_q = (1 - self.learning_rate) * old_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
    self.total_reward += reward

    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6
    self.q_table[self.state][self.action] = new_q

def calculate_reward(
    self,
    old_player_pos: Tuple[int, int],
    old_objective: Tuple[int, int],
    new_self_pos: Tuple[int, int],
    new_objective: Tuple[int, int],
    old_field: np.ndarray,
    new_field: np.ndarray,
    events: List[str]
) -> float:
    """
    Calculate the reward based on the state and events.

    :param old_player_pos: Previous position of the player.
    :param old_objective: Previous objective.
    :param new_self_pos: New position of the player.
    :param new_objective: New objective.
    :param old_field: Field representation before the action.
    :param new_field: Field representation after the action.
    :param events: List of events occurred.
    :return: Calculated reward.
    """
    old_distance = a.dijkstra(old_player_pos, old_objective, old_field)
    new_distance = a.dijkstra(new_self_pos, new_objective, new_field)

    if old_objective == new_objective:
        reward = 1 if len(new_distance) < len(old_distance) else -3
    else:
        reward = 0
    
    if "COIN_COLLECTED" in events:
        reward += 2
    
    return reward

def end_of_round(self, last_game_state: Dict, last_action: str, events: List[str]):
    """
    Handle end of round events and save relevant information.

    :param last_game_state: Final game state.
    :param last_action: Last action taken.
    :param events: List of end-of-round events.
    """
    reward = -100 if 'KILLED_SELF' in events else 0
    self.total_reward += reward

    self.graph.append((self.eps, self.total_reward))
    self.total_reward = 0

    if self.eps % 1000 == 0:
        self.save_metrics()
    
    self.eps += 1
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def save_metrics(self):
    """Save the graph and Q-table to files."""
    with open("graph.txt", "w") as file:
        json_dump(self.graph, file)
    with open("q_table.pickle", "wb") as file:
        pickle_dump(self.q_table, file)
