from collections import namedtuple, deque
import numpy as np
import pickle
from typing import List, Dict, Tuple
import events as e
from .callbacks import state_to_features

# Define a namedtuple for transitions (state, action, next_state, reward)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1000  # Increased history size for better learning
DISCOUNT_FACTOR = 0.99  # Gamma for future rewards
LEARNING_RATE = 0.1  # Alpha for Q-learning updates
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995

# Define possible actions
ACTIONS = ['Move Up', 'Move Down', 'Move Left', 'Move Right', 'Drop Bomb', 'Wait']

def setup_training(self):
    """
    Initialize the agent for training.
    This method is called after `setup` in callbacks.py.
    """
    # Setup a deque to store transitions with a limited history size
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # Initialize training parameters
    self.epsilon = EPSILON_START
    self.epsilon_min = EPSILON_MIN
    self.epsilon_decay = EPSILON_DECAY
    self.q_table = {}  # Use a dictionary for more flexible state-action pairs
    self.graph = []  # To store metrics (for example, rewards over episodes)
    self.total_reward = 0
    self.eps = 0

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.
    
    Update the agent's Q-table or policy based on game events that occurred during the step.
    """
    self.logger.debug(f'Encountered game event(s): {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Convert states to features using the provided helper function
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    
    # Calculate reward from events
    reward = reward_from_events(self, events)
    
    # Append the transition (state, action, reward, next_state) to the deque
    self.transitions.append(Transition(old_state, self_action, new_state, reward))
    
    # Update Q-table with the new transition
    update_q_table(self, old_state, self_action, reward, new_state)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to finalize rewards and update the model.
    
    This method is also responsible for saving the agent's model periodically.
    """
    self.logger.debug(f'Encountered event(s): {", ".join(map(repr, events))} in the final step')
    
    # Add the final transition with the last state and action
    last_state = state_to_features(last_game_state)
    final_reward = reward_from_events(self, events)
    self.transitions.append(Transition(last_state, last_action, None, final_reward))
    
    # Save the model periodically or after a set number of rounds
    if self.eps % 100 == 0:  # Example: save every 100 episodes
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.q_table, file)
    
    # Increment the episode count and update epsilon for exploration/exploitation
    self.eps += 1
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def reward_from_events(self, events: List[str]) -> int:
    """
    Calculate reward from the game events that occurred during a step.
    
    Customize rewards to encourage or discourage specific agent behaviors.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.CRATES_DESTROYED: 3,
        e.BOMB_DROPPED: -1,
        e.INVALID_ACTION: -1,
        e.KILLED_SELF: -100,
    }
    
    # Sum up the rewards from the events
    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def update_q_table(self, state, action, reward, next_state):
    """
    Update the Q-table using the Q-learning update rule.
    
    :param state: The current state in features.
    :param action: The action taken.
    :param reward: The reward received from the action.
    :param next_state: The next state after the action.
    """
    # Initialize Q-values for unseen states and actions
    if state not in self.q_table:
        self.q_table[state] = np.zeros(len(ACTIONS))
    if next_state not in self.q_table:
        self.q_table[next_state] = np.zeros(len(ACTIONS))
    
    # Get the current Q-value for the (state, action) pair
    current_q_value = self.q_table[state][ACTIONS.index(action)]
    
    # Find the max Q-value for the next state (future reward estimation)
    max_next_q_value = np.max(self.q_table[next_state])
    
    # Q-learning update rule
    self.q_table[state][ACTIONS.index(action)] = current_q_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * max_next_q_value - current_q_value
    )

def select_action(self, state):
    """
    Select an action based on the epsilon-greedy strategy.
    
    :param state: The current state in features.
    :return: The chosen action.
    """
    if np.random.rand() < self.epsilon:
        # Explore: Random action
        return np.random.choice(ACTIONS)
    else:
        # Exploit: Best action based on Q-table
        if state in self.q_table:
            return ACTIONS[np.argmax(self.q_table[state])]
        else:
            return np.random.choice(ACTIONS)  # Random action if state not seen

def log_training_progress(self, episode):
    """
    Log the agent's training progress, e.g., the average reward over recent episodes.
    
    :param episode: The current episode number.
    """
    # Example: Log the average reward over the last 100 episodes
    avg_reward = sum(self.graph[-100:]) / 100 if len(self.graph) >= 100 else sum(self.graph) / len(self.graph)
    self.logger.info(f"Episode {episode}: Average Reward: {avg_reward}")
