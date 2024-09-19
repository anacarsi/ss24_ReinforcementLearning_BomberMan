from .train import TRANSITION_HISTORY_SIZE,device,REQUIRE_HISTORY
import torch
import random
class History:
    def __init__(self, size,logger) -> None:
        self.state = torch.zeros((TRANSITION_HISTORY_SIZE, 5, 17, 17), device=device)
        self.action = torch.zeros((TRANSITION_HISTORY_SIZE,), device=device,dtype= torch.int64)
        self.next_state = torch.zeros((TRANSITION_HISTORY_SIZE, 5, 17, 17), device=device)
        self.reward = torch.zeros((TRANSITION_HISTORY_SIZE,), device=device)
        self.size = size
        self.index = 0  
        self.full_enough = False
        self.wrapped = 0
        self.logger = logger

    def append(self, state,action,next_state,reward):
        self.state[self.index] = state
        self.action[self.index] = action
        self.next_state[self.index] = next_state
        self.reward[self.index] = reward
        self.index += 1
        if self.index == self.size:
            self.index = 0
            self.wrapped +=1
            self.logger.info(f"HISTORY WAS written full {self.wrapped} times")

    def get_batch(self, amount: int) -> tuple | int:

        if self.wrapped >0:
            indices = torch.randperm(self.size)[:amount]
        elif self.index > REQUIRE_HISTORY:
            indices = torch.randperm(self.index)[:amount]
        else:
            if random.random() < 0.001:
                self.logger.info(f"history index was: {self.index} vs {REQUIRE_HISTORY} required")
            return self.index
        return (
            self.state[indices].to(device),
            self.action[indices].to(device, dtype=torch.int64),
            self.next_state[indices].to(device),
            self.reward[indices].to(device),
        )
