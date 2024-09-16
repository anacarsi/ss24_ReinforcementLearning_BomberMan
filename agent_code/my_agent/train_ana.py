# train.py
import numpy as np
import networkx as nx
import events as e
from json import dump as json_dump
from pickle import dump as pickle_dump
from typing import List, Dict, Tuple
from . import objective as a
from callbacks import state_to_features

def setup_training(self):
    """
    Called after setup in callbacks.py. This is used to set up the training environment.
    """
    # self.total_reward = 0
    self.episodes = 1000 # total training episodes
    self.max_steps = 100 # max steps per episode
    self.rewards = [] # store rewards
    self.graph = []

def game_events_occurred(
        self,
        old_game_state: Dict,
        action: str,
        new_game_state: Dict,
        events: List[str]
    ):
    """
    Handle game events and update the Q-table accordingly. Called once per step to allow intermediate rewards based on game events.

    :param old_game_state: The state before the action.
    :param self_action: The action taken.
    :param new_game_state: The state after the action.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`.
    """
    # Convert the game state into a feature vector
    state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    # Get the reward for the events
    reward = get_reward(events)

    # Update the Q-table
    update_q_table(state, action, reward, new_state)


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

def update_q_table(q_table, state, action, reward, next_state, alpha=0.1, gamma=0.98):
    """
    Update the Q-value for a given state-action pair using the Q-learning update rule.
    
    :param q_table: The Q-table storing Q-values.
    :param state: The current state.
    :param action: The action taken.
    :param reward: The reward received.
    :param next_state: The next state after taking the action.
    :param alpha: Learning rate.
    :param gamma: Discount factor.
    """
    # Find the max Q-value for the next state (future reward estimation)
    max_next_q_value = np.max(q_table[next_state])

    # Q-learning update rule
    current_q_value = q_table[state, action]
    q_table[state, action] = current_q_value + alpha * (reward + gamma * max_next_q_value - current_q_value)


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

#############################################

# alternative functions (TODO: implement)
def update_model(self, state, action, next_state, reward):
    """ Update the agent's model based on the collected experience """
    # Implement model update logic, for example using Q-learning:
    q_value = self.q_table[state][action]
    max_next_q_value = np.max(self.q_table[next_state])
    new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - q_value)
    self.q_table[state][action] = new_q_value

def log_training_progress(self, episode):
    """ Log the average reward over the last 100 episodes """
    avg_reward = sum(self.rewards[-100:]) / 100 if len(self.rewards) >= 100 else sum(self.rewards) / len(self.rewards)
    self.logger.info(f"Episode {episode}: Average Reward: {avg_reward}")

