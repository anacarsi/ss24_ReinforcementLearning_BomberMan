from collections import namedtuple, deque
import numpy as np
import json
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, create_vision

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
    Step by step reward calculation and Q-table update.

    This function uses the events that occurred between the old and new game state to adjust the agent's 
    rewards and Q-values. It also saves the experience (state-action-reward) in the replay buffer for future use.

    :param old_game_state: The game state before taking the action.
    :param self_action: The action the agent took.
    :param new_game_state: The resulting game state after taking the action.
    :param events: The list of events that occurred between the old and new game state.
    """

        # ------------------- NEW STATE CONSTRUCTION -------------------
    # Extract necessary information to build the new state representation
    field_representation, bombs, vision_field = state_to_features(self, new_game_state)  # Generate field map with bombs, crates, etc.
    can_place_bomb = new_game_state['self'][2] 
    player_x, player_y = new_game_state['self'][3] 
    
    # Agent's local vision around its position (for field and bomb map)
    vision_bomb_map = create_vision(self, bombs, (player_x, player_y), radius=4)

    # Create the new state representation (using bytes for efficient hashing)
    new_state = (vision_field.tobytes(), vision_bomb_map.tobytes(), can_place_bomb)

    # ------------------- REWARD CALCULATION -------------------
    # Every step has a small penalty to encourage efficient movement
    step_reward = -1
    
    # Reward adjustments based on game events
    if "CRATE_DESTROYED" in events:
        step_reward += 5  # Reward for destroying crates
    if "COIN_FOUND" in events:
        step_reward += 1  # Reward for discovering coins
    if "COIN_COLLECTED" in events:
        step_reward += 10  # Reward for collecting coins
    if "KILLED_OPPONENT" in events:
        step_reward += 10  # Reward for killing an opponent

    self.reward_episode += step_reward  # Update the total reward for the episode

    # ------------------- POTENTIAL-BASED REWARD SHAPING (PBRS) -------------------
    potential_psi = potential_function(self, self.reward_episode, self.max_reward_eps, self.min_reward_eps)

    # Save the current experience to the replay buffer
    self.replay_buffer.append((self.state, self.action, step_reward, potential_psi, new_state))

    # ------------------- Q-LEARNING UPDATE -------------------
    # Calculate the maximum future Q-value for the new state
    future_q_values = self.q_table.get(new_state, [-10] * 6)  # Get future state Q-values (initialize with -10 if state not seen)
    max_future_q = np.max(future_q_values)

    # Get the current Q-value for the state-action pair
    current_q_values = self.q_table.get(self.state, [-10] * 6)  # Get current state Q-values (initialize with -10 if state not seen)
    old_q_value = current_q_values[self.action]

    # Update the Q-value using the Q-learning formula: Q(s, a) ← (1 - α) * Q(s, a) + α * (r + γ * max_a' Q(s', a'))
    new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (step_reward + DISCOUNT_FACTOR * max_future_q)

    # If the state is not yet in the Q-table, initialize it with default Q-values
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6

    # Update the Q-value in the Q-table
    self.q_table[self.state][self.action] = new_q_value

    # ------------------- RESET FOR NEXT STEP -------------------
    # Update state and action for the next step in the next iteration
    self.state = new_state
    # self.action = self_action

    # ------------------- REPLAY -------------------
    # Store the transition in replay data
    transition_data = {
        'old_state': old_game_state,
        'action': self_action,
        'reward': step_reward,
        'new_state': new_game_state
    }
    self.replay_data.append(transition_data)

def potential_function(self, reward_episode, max_reward_eps, min_reward_eps):
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
    
    Parameters:
    - last_game_state (dict): The final game state of the episode.
    - last_action (str): The last action taken by the agent.
    - events (List[str]): The list of events that occurred during the last game state.
    """

    # ------------------- REWARD CALCULATION -------------------
    reward = 0
    if e.SURVIVED_ROUND in events:
        reward += 100
    # if e.COLLECTED_EVERYTHING in events and e.SURVIVED_ROUND in events:
        # reward += 100
    if e.SURVIVED_ROUND in events:
        reward += 100
    if e.COIN_COLLECTED in events:
        reward += 10
    if e.KILLED_SELF in events:
        reward -= 100
    if e.GOT_KILLED in events:
        reward -= 100

    self.reward_episode += reward  # Accumulate the total reward for this episode

    # ------------------- Q-LEARNING WITH PBRS -------------------
    # Iterate backward through the replay buffer to apply potential-based reward shaping
    for i in range(len(self.replay_buffer) - 2, -1, -1):
        (state, action, _, psi_s, next_state) = self.replay_buffer[i]
        (_, _, _, next_psi_s, _) = self.replay_buffer[i + 1]
        
        # Update the Q-values in the replay buffer using a potential difference (new_psi_s - psi_s) to shape rewards:
        potential_difference = DISCOUNT_FACTOR * next_psi_s - psi_s
        self.q_table[state][action] += LEARNING_RATE * potential_difference

    # ------------------- TRACK MAXIMUM/MINIMUM REWARD -------------------
    # Update the max/min reward for the episode, only if reward_episode is non-zero
    if self.reward_episode > self.max_reward_eps and self.reward_episode > 0:
        self.max_reward_eps = self.reward_episode
    elif self.reward_episode < self.min_reward_eps and self.reward_episode < 0:
        self.min_reward_eps = self.reward_episode

    # ------------------- UPDATE Q-TABLE FOR FINAL STATE -------------------
    # Initialize state-action values in Q-table if the state is unseen
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6  # Initialize Q-values to -10 for each action
    
    # Update Q-value of the final action using reward
    self.q_table[self.state][self.action] += LEARNING_RATE * reward

    # ------------------- SAVE Q-TABLE PERIODICALLY -------------------
    # Every 1000 episodes, save the Q-table to a file
    if self.eps % 1000 == 0:
        with open("q_table.pickle", "wb") as file:
            pickle.dump(self.q_table, file)

    # ------------------- EPSILON (EXPLORATION RATE) MANAGEMENT -------------------
    # Reduce epsilon (exploration rate) every 100 episodes
    if self.eps % 100 == 0:
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # Every 10,000 episodes, reset epsilon to 0.3 for more exploration
    if self.eps % 10000 == 0:
        self.epsilon = 0.3

    # ------------------- RESET EPISODE VARIABLES -------------------
    self.eps += 1  # Increment the episode counter
    self.reward_episode = 0  # Reset the episode reward
    self.replay_buffer.clear()  # Clear the replay buffer for the next episode

    # Save the replay data at the end of the episode
    save_replay(self.replay_data)
    # Reset the replay data for the next episode
    self.replay_data.clear()


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
    
    Customize rewards to encourage or discourage specific agent behaviors.
    """
    game_rewards = {
        e.SURVIVED_ROUND: 1,
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

def log_training_progress(self, episode):
    """
    Log the agent's training progress, e.g., the average reward over recent episodes.
    
    :param episode: The current episode number.
    """
    # Example: Log the average reward over the last 100 episodes
    avg_reward = sum(self.graph[-100:]) / 100 if len(self.graph) >= 100 else sum(self.graph) / len(self.graph)
    self.logger.info(f"Episode {episode}: Average Reward: {avg_reward}")
