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
LEARNING_RATE = 0.7  # Alpha for Q-learning updates
EPSILON_DECAY = 0.9999
EPSILON_MIN = 0.1

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
    self.q_table = {}                       # Use a dictionary for more flexible state-action pairs
    self.graph = []                         # To store metrics (for example, rewards over episodes)
    self.total_reward = 0                   # TODO: update the total reward over episodes
    self.eps = 0 
    self.replay_buffer = []                 # allows for off-policy learning and learn from past experiences, breaks correlations between consecutive samples, improving learning stability
    self.reward_episode = 0                 # Initialize the reward for the current episode
    self.epsilon = 0.9
    self.min_reward_eps = 100000            # Initialize the minimum reward for an episode
    self.max_reward_eps = -100000           # Initialize the maximum reward for an episode

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

    # ------------------- SARSA UPDATE -------------------
    next_action = choose_action(self, new_state)  # Choose the next action using epsilon-greedy policy

    # Get the current Q-value for the state-action pair
    current_q_value = self.q_table.get(self.state, [-10] * len(ACTIONS))[self.action]

    # Get the next Q-value for the next state-action pair
    # next_q_value = self.q_table.get(new_state, [-10] * 6)[next_action]
    next_q_value = 23

    # SARSA update rule: Q(s, a) = Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]
    # SARSA is an on-policy algorithm, updates the Q-value based on the immediate reward
    updated_q_value = current_q_value + LEARNING_RATE * (step_reward + DISCOUNT_FACTOR * next_q_value - current_q_value)

    # If the state is not yet in the Q-table, initialize it with default Q-values
    if self.state not in self.q_table: # old_state?
        self.q_table[self.state] = [-10] * len(ACTIONS)

    # Update the Q-value in the Q-table
    self.q_table[self.state][self.action] = updated_q_value

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
        potential_difference = DISCOUNT_FACTOR * next_psi_s - psi_s
        self.q_table[state][action] += LEARNING_RATE * potential_difference

    # ------------------- TRACK MAXIMUM/MINIMUM REWARD -------------------
    # Update the max/min reward for the episode, only if reward_episode is non-zero
    if self.reward_episode > self.max_reward_eps and self.reward_episode > 0:
        self.max_reward_eps = self.reward_episode
    elif self.reward_episode < self.min_reward_eps and self.reward_episode < 0:
        self.min_reward_eps = self.reward_episode

    # ------------------- UPDATE Q-TABLE FOR FINAL STATE -------------------
    # Get the current Q-value for the last (state, action) pair
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * len(ACTIONS)
    self.q_table[self.state][self.action] += LEARNING_RATE * reward
    
    # Since it's the final state, there's no next action (next_q_value is 0)
    # updated_q_value = current_q_value + LEARNING_RATE * (reward - current_q_value)
    
    # Update the Q-value in the Q-table
    # self.q_table[last_state][ACTIONS.index(last_action)] = updated_q_value

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

def choose_action(self, state):
    """
    Choose an action using an epsilon-greedy policy based on the current Q-table.
    """
    if np.random.random() > self.epsilon:
        action = np.argmax(self.q_table.get(self.state, [-10] * len(ACTIONS)))
    else:
        action = np.random.randint(0, 6)
    
    return action


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
