import torch
import random
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
TRANSITION_HISTORY_SIZE = 800_000  # keep only 1M last transitions
REQUIRE_HISTORY = 100_000  # 100k
class History:
    def __init__(self, size,logger) -> None:
        self.state = torch.zeros((TRANSITION_HISTORY_SIZE, 880), device=device)
        self.action = torch.zeros((TRANSITION_HISTORY_SIZE,), device="cpu",dtype= torch.int64)
        self.next_state = torch.zeros((TRANSITION_HISTORY_SIZE, 880), device=device)
        self.reward = torch.zeros((TRANSITION_HISTORY_SIZE,), device=device)
        self.models = torch.zeros((TRANSITION_HISTORY_SIZE,2),device="cpu",dtype=torch.int32)
        self.size = size
        self.index = 0  
        self.full_enough = False
        self.wrapped = 0
        self.logger = logger

    def append(self, state,action,next_state,reward,models:tuple[int,int]):
        self.state[self.index] = state
        self.action[self.index] = action
        self.next_state[self.index] = next_state
        self.reward[self.index] = reward
        self.models[self.index]= torch.tensor(models,dtype=torch.int32)
        self.index += 1
        if self.index == self.size:
            self.index = 0
            self.wrapped +=1
            self.logger.info(f"HISTORY WAS written full {self.wrapped} times")

    def get_one(self):
        if self.wrapped > 0:
            index = random.randint(0,self.size-1)
        elif self.index >REQUIRE_HISTORY:
            index = random.randint(0,self.index-1)
        else:
            return self.index
        return (self.state[index],self.action[index],self.next_state[index],self.reward[index],self.models[index])
    
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
            self.action[indices].to(dtype=torch.int64),
            self.next_state[indices].to(device),
            self.reward[indices].to(device),
        )
