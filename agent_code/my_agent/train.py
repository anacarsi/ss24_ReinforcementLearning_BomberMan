from collections import namedtuple, deque
import pickle
from typing import List

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 1000
RECORD_ENEMY_TRANSITIONS = 1.0

def setup_training(self):
    """
    Initialize self for Q-learning training.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Handle game events and update Q-values.
    """
    state = state_to_features(old_game_state)
    next_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    self.transitions.append(Transition(state, self_action, next_state, reward))

    if len(self.transitions) >= 1:
        # Sample a transition
        transition = self.transitions[-1]
        self._update_q_values(transition)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Handle end-of-round events and save the Q-table.
    """
    state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    
    # Final reward processing
    self.transitions.append(Transition(state, last_action, None, reward))
    
    for transition in self.transitions:
        self._update_q_values(transition)

    # Save the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Convert game events to rewards.
    """
    game_rewards = {
        'COIN_COLLECTED': 1,
        'KILLED_OPPONENT': 5,
        'PLACEHOLDER_EVENT': -0.1
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def _update_q_values(self, transition: Transition):
    """
    Update Q-values based on the observed transition.
    """
    state, action, next_state, reward = transition
    
    # Convert states and actions to indices
    state_key = tuple(state)
    action_index = ACTIONS.index(action)
    
    # Get current Q-values
    current_q_values = self.q_table.get(state_key, np.zeros(NUM_ACTIONS))
    
    # Get max Q-value for next state
    next_q_values = self.q_table.get(tuple(next_state), np.zeros(NUM_ACTIONS))
    max_next_q_value = np.max(next_q_values)
    
    # Update Q-value
    new_q_value = current_q_values[action_index] + self.alpha * (reward + self.gamma * max_next_q_value - current_q_values[action_index])
    
    # Save updated Q-value
    current_q_values[action_index] = new_q_value
    self.q_table[state_key] = current_q_values
