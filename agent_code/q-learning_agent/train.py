from pickle import dump as pickle_dump
from typing import List
import numpy as np
import events as e
from json import dump as json_dump
from . import auxiliary_functions as a
from . import hyperparameters as hyp

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #current episode
    self.eps = 0
    #maximal episode reward
    self.maxreward_episode = -10000
    #minimal episode reward
    self.minreward_episode = 100000
    #total reward of current episode
    self.reward_episode = 0
    #initialize replay buffer
    self.replay_buffer = []


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
    
    #BUILD NEW FIELD AND NEW STATE
    new_field = a.build_field(self, new_game_state)
    new_can_place_bomb = new_game_state['self'][2]
    new_expl_map = a.bomb_map(self, new_game_state)
    new_player_pos_x, new_player_pos_y = new_game_state['self'][3]
    new_vision1 = a.agent_vision(self, new_field, (new_player_pos_x, new_player_pos_y), hyp.VISION_RANGE)
    new_vision2 = a.agent_vision(self, new_expl_map, (new_player_pos_x, new_player_pos_y), hyp.VISION_RANGE)

    new_state = (new_vision1.tobytes(), new_vision2.tobytes(), new_can_place_bomb)

    #REWARD CALCULATION (STEP)
    #every step is penalized, therefore it should try to take shortest path.
    reward = -1
    if "CRATE_DESTROYED" in events:
        reward += 5
    if "COIN_FOUND" in events:
        reward += 1
    if "COIN_COLLECTED" in events:
        reward += 10
    if "KILLED_OPPONENT" in events:
        reward += 10
    self.reward_episode += reward

    #Q-LEARNING ALGORITHM (STEP)
    self.reward_episode = 0
    if reward == 0:
        psi_s = 0
    else:
        #print(self.reward_episode, self.minreward_episode, self.maxreward_episode)
        psi_s = 1 + (self.reward_episode - self.maxreward_episode) / (self.maxreward_episode - self.minreward_episode)
    self.replay_buffer.append((self.state, self.action, reward, psi_s, new_state))

    #UPDATE Q-TABLE
    max_future_q = np.max(self.q_table.get(new_state, [-10] * 6))
    old_q = self.q_table.get(self.state, [-10] * 6)[self.action]
    #q function (returning qvalue for the current state and action)
    new_q = (1 - self.learning_rate) * old_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6
        # print("State not in q_table training")
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
    if e.SURVIVED_ROUND in events:
        reward += 100    
    if (e.COLLECTED_EVERYTHING in events) and (e.SURVIVED_ROUND in events):
        reward += 100
    if e.COIN_COLLECTED in events:
        reward += 10
    if e.KILLED_SELF in events:
        reward += -100
    if e.GOT_KILLED in events:
        reward += -100
    self.reward_episode += reward

    #Q LEARNING + POTENTIAL BASED REWARD SHAPING METHOD
    #From len(replay_buffer)-2 downto 0
    for i in range(len(self.replay_buffer) - 2, -1, -1): 
        (s,a,r,psi_s,s_prima) = self.replay_buffer[i]
        (new_s,new_a,new_r,new_psi_s,s_prima_prima) = self.replay_buffer[i+1]
        potential_function = self.discount_factor * new_psi_s - psi_s
        self.q_table[s][a] += self.learning_rate * potential_function

    if self.reward_episode > self.maxreward_episode & self.reward_episode > 0:
        self.maxreward_episode = self.reward_episode
    elif self.reward_episode < self.minreward_episode & self.reward_episode < 0:
        self.minreward_episode = self.reward_episode

    #UPDATE Q-TABLE AND Q-FUNCTION
    if self.state not in self.q_table:
        self.q_table[self.state] = [-10] * 6
    self.q_table[self.state][self.action] += self.learning_rate * reward

    
    if self.eps % 1000 == 0:
        # Store the Q table
        with open("q_table.pickle", "wb") as file:
            pickle_dump(self.q_table, file)
    
    
    if self.eps % 100 == 0:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    if self.eps % 10000 == 0:
        self.epsilon = 0.3

    self.eps += 1
    self.reward_episode = 0
    self.replay_buffer = []
