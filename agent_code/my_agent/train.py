from collections import namedtuple, deque
import numpy as np
import pickle
from typing import List
import events as e
from .callbacks import state_to_features, create_vision

# Define a namedtuple for transitions (state, action, next_state, reward)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
TRANSITION_HISTORY_SIZE = 1000  # Increased history size for better learning
DISCOUNT_FACTOR = 0.99  # Gamma for future rewards
LEARNING_RATE = 0.1  # Alpha for Q-learning updates
MIN_REWARD_EPISODE = 100000
MAX_REWARD_EPISODE = -100000
EPSILON_DECAY = 0.9999

# Define possible actions
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialize the agent for training. Potential-Based Reward Shaping initialization: maxreward_episode = -inf, minreward_episode = inf, reward_episode = 0.
    This method is called after `setup` in callbacks.py.
    """
    # Setup a deque to store transitions with a limited history size
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    # Initialize training parameters
    self.min_reward_eps = MIN_REWARD_EPISODE
    self.max_reward_eps = MAX_REWARD_EPISODE
    self.epsilon_decay = EPSILON_DECAY
    self.q_table = {}                       # Use a dictionary for more flexible state-action pairs
    self.graph = []                         # To store metrics (for example, rewards over episodes)
    self.total_reward = 0                   # TODO: update the total reward over episodes
    self.eps = 0 
    self.replay_buffer = []                 # allows for off-policy learning and learn from past experiences, breaks correlations between consecutive samples, improving learning stability
    self.reward_episode = 0                 # Initialize the reward for the current episode
    self.learning_rate = 0.7
    self.discount_factor = 0.8
    self.epsilon_min = 0.1
    self.epsilon = 0.9
 
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
    # The potential function (psi) adjusts the reward based on the current episode's progress.
    # If no reward was earned (step_reward == 0), psi_s is set to 0, otherwise, it's shaped by past rewards.
    if step_reward == 0:
        potential_psi = 0
    else:
        # Normalize potential reward shaping using episode's min/max rewards
        if self.max_reward_eps - self.min_reward_eps != 0:
            potential_psi = 1 + (self.reward_episode - self.max_reward_eps) / (self.max_reward_eps - self.min_reward_eps)
        else:
            potential_psi = 1  # Avoid division by zero in the first few episodes

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
    new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (step_reward + self.discount_factor * max_future_q)

    # If the state is not yet in the Q-table, initialize it with default Q-values
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6

    # Update the Q-value in the Q-table
    self.q_table[self.state][self.action] = new_q_value

    # ------------------- RESET FOR NEXT STEP -------------------
    # Update state and action for the next step in the next iteration
    self.state = new_state

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
        potential_difference = self.discount_factor * next_psi_s - psi_s
        self.q_table[state][action] += self.learning_rate * potential_difference

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
    self.q_table[self.state][self.action] += self.learning_rate * reward

    # ------------------- SAVE Q-TABLE PERIODICALLY -------------------
    # Every 1000 episodes, save the Q-table to a file
    if self.eps % 1000 == 0:
        with open("q_table.pickle", "wb") as file:
            pickle.dump(self.q_table, file)

    # ------------------- EPSILON (EXPLORATION RATE) MANAGEMENT -------------------
    # Reduce epsilon (exploration rate) every 100 episodes
    if self.eps % 100 == 0:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Every 10,000 episodes, reset epsilon to 0.3 for more exploration
    if self.eps % 10000 == 0:
        self.epsilon = 0.3

    # ------------------- RESET EPISODE VARIABLES -------------------
    self.eps += 1  # Increment the episode counter
    self.reward_episode = 0  # Reset the episode reward
    self.replay_buffer.clear()  # Clear the replay buffer for the next episode

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
