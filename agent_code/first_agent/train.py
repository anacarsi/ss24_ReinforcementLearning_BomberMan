from collections import namedtuple, deque

import pickle
from typing import List
import torch
import torch.optim as optim
import random
import events as e
from .callbacks import state_to_features
from datetime import datetime,timedelta
# This is only an example!
from numpy import ndarray as arr
import numpy as np
from logging import Logger
BATCH_SIZE = 1000
EPISODES = 10_000
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

AMOUNT_FEATURES = 372

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class History():
    def __init__(self,size) -> None:
        self.state= torch.zeros((TRANSITION_HISTORY_SIZE,AMOUNT_FEATURES))
        self.action = torch.zeros((TRANSITION_HISTORY_SIZE,))
        self.next_state = torch.zeros((TRANSITION_HISTORY_SIZE,AMOUNT_FEATURES))
        self.reward = torch.zeros((TRANSITION_HISTORY_SIZE,))
        self.size = size
        self.index = 0
        self.full_enough = False
        self.wrapped = False
    def append(self,item:Transition):
        self.state[self.index]=item.state
        self.action[self.index]=item.action
        self.next_state[self.index]=item.next_state
        self.reward[self.index]=item.reward
        self.index+=1
        if self.index == self.size:
            self.index = 0
            self.wrapped = True

    def get_batch(self,amount:int):
        
        if self.wrapped == True:
            indices= torch.randperm(self.size)[:amount]
        if amount >= self.index:
            return None
        else:
            indices = torch.randperm(self.index-1)[:amount]
        return (self.state[indices].to(device),self.action[indices].to(device,dtype=torch.int64),
         self.next_state[indices].to(device),self.reward[indices].to(device))

class DummySelf():
    transitions: History
    logger:Logger
    epsilon:float
    model:torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_method: torch.nn.modules.loss._Loss
    last_saved:datetime
    loss:list[float]
    gamma:float




# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 30000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 0.01  # record enemy transitions with probability ...



# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self:DummySelf):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.transitions = History(TRANSITION_HISTORY_SIZE)

    self.epsilon = 0.02
    self.gamma = 0.98
    self.loss_method = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(),0.001)
    self.last_saved = datetime.now()
    self.loss=[]

def game_events_occurred(self:DummySelf, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
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
    #self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state)[0], ACTIONS.index(self_action), state_to_features(new_game_state)[0], reward_from_events(self, events)))


def end_of_round(self:DummySelf, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state)[0], ACTIONS.index(last_action), torch.full((AMOUNT_FEATURES,),torch.nan), reward_from_events(self, events)))
    
    
    # try to train:
    #minibatch = random.sample(self.transitions,BATCH_SIZE)

    minibatch = self.transitions.get_batch(BATCH_SIZE)
    if minibatch is None:
        self.logger.debug(f"sampling failed, likely due to not enough experience. The exception was: {e}")
        return # ignore error
    state,action,next_state,reward = minibatch
    
    
    yi_arr = torch.zeros(BATCH_SIZE) # store the mse. #TODO: dont store them, compute average in one go.
    q_value = torch.zeros(BATCH_SIZE)

 

    q_value :torch.tensor= self.model.forward(state)
    with torch.no_grad():
        target_q:torch.tensor = self.model.forward(next_state)
    target_q_mask :torch.tensor= ~target_q[:,0].isnan()
    q_value = q_value.gather(1,action.unsqueeze(1)).squeeze()
    target_q = target_q.max(dim=1).values
    yi_arr = reward + self.gamma * (target_q_mask * target_q).nan_to_num(0.0) 

    loss = self.loss_method(q_value,yi_arr)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    self.loss.append(loss.item())

    self.logger.debug(f"Loss is: {loss.item()}")
    # Store the model
    now = datetime.now()
    if now - self.last_saved > timedelta(minutes=1):
        torch.save(self.model.state_dict(),f"snapshot_model_{now.isoformat()}.pth")
        self.last_saved = now
        self.logger.info("saved snapshot")
    if last_game_state["round"]==EPISODES:
        now =datetime.now().isoformat()
        
        torch.save(self.model.state_dict(),f"finished_train_snapshot_model_{now}.pth")
        torch.save(self.model.state_dict(),f"model.pth")
        
        np.save(f"loss_history_{now}.npy",np.array(self.loss))
    self.logger.debug(f"Score at end of round:  {last_game_state["self"][1]}")

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1, # official 
        e.KILLED_OPPONENT: 5, # official 
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    #self.logger.debug(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
