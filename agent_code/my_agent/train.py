from collections import namedtuple, deque
import numpy as np
import json
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, Features

# Define a namedtuple for transitions (state, action, next_state, reward)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1000  # Increased history size for better learning
DISCOUNT_FACTOR = 0.99  # Gamma for future rewards
LEARNING_RATE = 0.7  # Alpha for Q-learning updates
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.1

# Define possible actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialize the agent for training with SARSA.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)  # Store state-action transitions
    self.q_table = {}                       # Use a dictionary for more flexible state-action pairs
    self.graph = []                         # To store metrics (for example, rewards over episodes)
    self.total_reward = 0                   # TODO: update the total reward over episodes
    self.eps = 0 
    self.replay_buffer = []                 # Allows for off-policy learning and learn from past experiences, breaks correlations between consecutive samples, improving learning stability
    self.reward_episode = 0                 # Sum of rewards in current episode
    self.epsilon = 0.9
    self.min_reward_eps = 100000            # Minimum value of episode until now
    self.max_reward_eps = -100000           # Maximum value of episode until now
    self.replay_data = []                   # Store the replay data for training

 
def game_events_occurred_old(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    
    Update the agent's Q-table or policy based on game events that occurred during the step.
    """
    self.logger.debug(f'Encountered game event(s): {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Convert states to features using the provided helper function
    old_state = state_to_features(self, old_game_state)
    new_state = state_to_features(self, new_game_state)
    
    # Calculate reward from events
    reward = reward_from_events(self, events)
    
    # Append the transition (state, action, reward, next_state) to the deque
    self.transitions.append(Transition(old_state, self_action, new_state, reward))
    
    # Update Q-table with the new transition
    update_q_table(self, old_state, self_action, reward, new_state)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Step-by-step reward calculation and Q-table update.
    
    This function updates the Q-values based on the events that occurred between the old and new game states, and saves the
    experience (state-action-reward-new_state) for future replay.

    :param old_game_state: The game state before the action was taken.
    :param self_action: The action the agent took.
    :param new_game_state: The game state after the action was taken.
    :param events: The list of events that occurred between the old and new game state.
    """
    
    # ------------------- NEW STATE CONSTRUCTION -------------------
    # Extract features from the new game state
    if new_game_state is not None:
        new_features = state_to_features(new_game_state)
        # Use the relevant features as the new state representation
        new_state = (
            new_features.x_bomb, new_features.y_bomb, 
            new_features.x_coin, new_features.y_coin, 
            new_features.x_crate, new_features.y_crate, 
            new_features.safe_down, new_features.safe_up, 
            new_features.safe_left, new_features.safe_right, 
            new_features.safe_stay, new_features.possible_down, 
            new_features.possible_up, new_features.possible_right, 
            new_features.possible_left, new_features.can_place_bomb
        )
    else:
        new_state = None  # If the game ends, there may not be a new state

    # ------------------- OLD STATE CONSTRUCTION -------------------
    # Extract features from the old game state
    if old_game_state is not None:
        old_features = state_to_features(old_game_state)
        old_state = (
            old_features.x_bomb, old_features.y_bomb, 
            old_features.x_coin, old_features.y_coin, 
            old_features.x_crate, old_features.y_crate, 
            old_features.safe_down, old_features.safe_up, 
            old_features.safe_left, old_features.safe_right, 
            old_features.safe_stay, old_features.possible_down, 
            old_features.possible_up, old_features.possible_right, 
            old_features.possible_left, old_features.can_place_bomb
        )
    else:
        return  # If there is no old state (like at the start of the game), there's nothing to do

    # ------------------- REWARD CALCULATION -------------------
    # Default step reward (negative to encourage quicker decision-making)
    step_reward = -1

    # Event-based reward adjustments
    if "CRATE_DESTROYED" in events:
        step_reward += 5  # Destroying crates gives a reward
    if "COIN_FOUND" in events:
        step_reward += 1  # Finding coins is rewarded
    if "COIN_COLLECTED" in events:
        step_reward += 10  # Collecting coins gives a large reward
    if "KILLED_OPPONENT" in events:
        step_reward += 10  # Killing an opponent gives a large reward

    # Penalizing getting caught in explosions or dying
    if "GOT_KILLED" in events or "KILLED_SELF" in events:
        step_reward -= 50  # Severe penalty for dying

    # Update the episode's total reward
    self.reward_episode += step_reward

    # ------------------- Q-LEARNING UPDATE -------------------
    # Get the current Q-value for the old state-action pair
    if old_state in self.q_table:
        current_q_values = self.q_table[old_state]
    else:
        # Initialize Q-values for unseen states
        current_q_values = [-10] * len(ACTIONS)
    
    old_q_value = current_q_values[self.action]

    # Calculate the maximum future Q-value for the new state
    if new_state is not None:
        if new_state in self.q_table:
            future_q_values = self.q_table[new_state]
        else:
            # If the new state hasn't been encountered, initialize Q-values
            future_q_values = [-10] * len(ACTIONS)
        max_future_q = np.max(future_q_values)
    else:
        max_future_q = 0  # No future Q-value if the game ended (e.g., agent died or game finished)

    # Update the Q-value using the Q-learning formula
    new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (step_reward + DISCOUNT_FACTOR * max_future_q)

    # Update the Q-table with the new Q-value for the state-action pair
    current_q_values[self.action] = new_q_value
    self.q_table[old_state] = current_q_values

    # ------------------- SAVE EXPERIENCE FOR REPLAY -------------------
    # Append the experience to the replay buffer
    self.replay_buffer.append((old_state, self.action, step_reward, new_state))

    # Update the current state for the next iteration
    self.state = new_state


def potential_function(self, reward_episode: int, max_reward_eps: int, min_reward_eps: int):
    """
    Calculate the potential-based reward shaping function psi_s.
    
    :param reward_episode: The total reward accumulated in the current episode.
    :param max_reward_eps: The maximum reward achieved in any episode.
    :param min_reward_eps: The minimum reward achieved in any episode.
    :return: The potential-based reward shaping function psi_s.
    """
    if reward_episode == 0:
        return 0
    elif max_reward_eps - min_reward_eps != 0:
        return 1 + (reward_episode - max_reward_eps) / (max_reward_eps - min_reward_eps)
    else:
        return 1

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent dies to calculate rewards, update the Q-table, and manage exploration.
    """

    # ------------------- REWARD CALCULATION -------------------
    reward = reward_from_events(self, events)

    # Final state representation from features
    if last_game_state is not None:
        last_features = state_to_features(last_game_state)
        last_state = (
            last_features.x_bomb, last_features.y_bomb, 
            last_features.x_coin, last_features.y_coin, 
            last_features.x_crate, last_features.y_crate, 
            last_features.safe_down, last_features.safe_up, 
            last_features.safe_left, last_features.safe_right, 
            last_features.safe_stay, last_features.possible_down, 
            last_features.possible_up, last_features.possible_right, 
            last_features.possible_left, last_features.can_place_bomb
        )
    else:
        last_state = None  # No state if game ends abruptly

    # Update the Q-value for the last state and action
    update_q_table(self, last_state, last_action, reward, None)  # Next state is None (game over)

    # ------------------- RESET EPISODE VARIABLES -------------------
    self.eps += 1  # Increment the episode counter
    self.reward_episode = 0  # Reset the total reward for the episode
    self.replay_buffer.clear()  # Clear the replay buffer for the next round

    if self.eps % 1000 == 0:
        with open("q_table.pickle", "wb") as file:
            pickle.dump(self.q_table, file)

    # Decrease exploration rate every 100 episodes
    if self.eps % 100 == 0:
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    if self.eps % 10000 == 0:
        self.epsilon = 0.3

    save_replay(self.replay_data)
    self.replay_data.clear()  # Clear for next episode

def save_replay(replay_data, filename="replay.json"):
    """
    Save replay data to a JSON file.
    
    :param replay_data: The replay data containing game states and transitions.
    :param filename: The name of the file to save the replay data.
    """
    with open(filename, 'w') as f:
        json.dump(replay_data, f)


def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate reward from the game events that occurred during a step.
    """
    game_rewards = {
        e.SURVIVED_ROUND: 100,
        e.KILLED_OPPONENT: 10,
        e.CRATE_DESTROYED: 5,
        e.COIN_COLLECTED: 10,
        e.GOT_KILLED: -100,
        e.KILLED_SELF: -100,
        e.INVALID_ACTION: -1,
    }

    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def update_q_table(self, state, action, reward, next_state):
    """
    Update the Q-table using the Q-learning update rule.
    
    :param state: The current state represented by extracted features.
    :param action: The action taken.
    :param reward: The reward received from the action.
    :param next_state: The next state after the action.
    """
    # Initialize Q-values for unseen states
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    if next_state is not None and next_state not in self.q_table:
        self.q_table[next_state] = np.zeros(len(ACTIONS))

    # Current Q-value for (state, action)
    current_q_value = self.q_table[state][ACTIONS.index(action)]
    
    # Find the max Q-value for the next state
    if next_state is not None:
        max_next_q_value = np.max(self.q_table[next_state])
    else:
        max_next_q_value = 0  # No future if the game ended
    
    # Q-learning update rule
    self.q_table[state][ACTIONS.index(action)] = current_q_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * max_next_q_value - current_q_value
    )


def log_training_progress(self, episode):
    """
    Log the agent's training progress, e.g., the average reward over recent episodes.
    
    :param episode: The current episode number.
    """
    # Example: Log the average reward over the last 100 episodes
    avg_reward = sum(self.graph[-100:]) / 100 if len(self.graph) >= 100 else sum(self.graph) / len(self.graph)
    self.logger.info(f"Episode {episode}: Average Reward: {avg_reward}")
