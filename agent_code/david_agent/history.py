from .train import TRANSITION_HISTORY_SIZE, device, REQUIRE_HISTORY
import torch
import random
import numpy as np

ALPHA = 0.6
BETA = 1


class History:
    def __init__(self, size, logger) -> None:
        self.state = torch.zeros((TRANSITION_HISTORY_SIZE, 5, 17, 17), device=device)
        self.action = torch.zeros((TRANSITION_HISTORY_SIZE,), device=device, dtype=torch.int64)
        self.next_state = torch.zeros((TRANSITION_HISTORY_SIZE, 5, 17, 17), device=device)
        self.reward = torch.zeros((TRANSITION_HISTORY_SIZE,), device=device)

        # K-step theory:

        # old:
        # yi_arr = reward + self.gamma * (target_q).nan_to_num(0.0)

        # yi_arr =reward1 + gamma * reward2 +  gamma^k * reward..k
        # reward1 + gamma * reward2 +  gamma^k can already be stored in the history reward array.
        #
        # self.gamma_k = torch.zeros(...)

        # during training
        # yi_arr = reward_of_k_next_steps_discounted + self.gamma^k+1 * max(Q_val(next_step))
        # not sure about the indices, one by off errors

        self.size = size
        self.index = 0
        self.full_enough = False
        self.wrapped = 0
        self.logger = logger

    def append(self, state, action, next_state, reward):
        self.state[self.index] = state
        self.action[self.index] = action
        self.next_state[self.index] = next_state
        self.reward[self.index] = reward
        self.index += 1
        if self.index == self.size:
            self.index = 0
            self.wrapped += 1
            self.logger.info(f"HISTORY WAS written full {self.wrapped} times")

    def get_batch(self, amount: int) -> tuple | int:

        if self.wrapped > 0:
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


class HistoryPrioritized:
    def __init__(self, size, logger) -> None:
        self.state = torch.zeros((TRANSITION_HISTORY_SIZE, 5, 17, 17), device=device)
        self.action = torch.zeros((TRANSITION_HISTORY_SIZE,), device=device, dtype=torch.int64)
        self.next_state = torch.zeros((TRANSITION_HISTORY_SIZE, 5, 17, 17), device=device)
        self.reward = torch.zeros((TRANSITION_HISTORY_SIZE,), device=device)
        self.error = torch.zeros((TRANSITION_HISTORY_SIZE,), dtype=torch.float32,device=device)
        # K-step theory:

        # old:
        # yi_arr = reward + self.gamma * (target_q).nan_to_num(0.0)

        # yi_arr =reward1 + gamma * reward2 +  gamma^k * reward..k
        # reward1 + gamma * reward2 +  gamma^k can already be stored in the history reward array.
        #
        # self.gamma_k = torch.zeros(...)

        # during training
        # yi_arr = reward_of_k_next_steps_discounted + self.gamma^k+1 * max(Q_val(next_step))
        # not sure about the indices, one by off errors
        self.size = size
        self.index = 0
        self.full_enough = False
        self.wrapped = 0
        self.max_err = 1
        self.logger = logger

    def append(self, state, action, next_state, reward):
        self.state[self.index] = state
        self.action[self.index] = action
        self.next_state[self.index] = next_state
        self.reward[self.index] = reward
        self.error[self.index] = self.max_err
        self.index += 1
        if self.index == self.size:
            self.index = 0
            self.wrapped += 1
            self.logger.info(f"HISTORY WAS written full {self.wrapped} times")

    def get_batch(self, amount: int) -> tuple | int:
        """
        returns an int if not ready yet,
        otherwise returns minibatch as tuple
        """
        if self.wrapped > 0:
            probabilites = (self.error + 1e-3) ** ALPHA
            probabilites = probabilites / probabilites.sum()
            indices = np.random.choice(self.size, amount, replace=True, p=probabilites.cpu().numpy())
            weights = ((1 / probabilites[indices]) / TRANSITION_HISTORY_SIZE) ** BETA

            weights = weights / weights.max(0).values
        elif self.index > REQUIRE_HISTORY:
            # indices = torch.randperm(self.index)[:amount]
            probabilites = (self.error[: self.index] + 1e-4) ** ALPHA
            probabilites = probabilites / probabilites.sum()
            indices = np.random.choice(self.index, amount, replace=True, p=probabilites.cpu().numpy())
            weights = ((1 / probabilites[indices]) / self.index) ** BETA
            weights = weights / weights.max(0).values

        else:
            if random.random() < 0.001:
                self.logger.info(f"history index was: {self.index} vs {REQUIRE_HISTORY} required")
            return self.index
        return (
            self.state[indices].to(device),
            self.action[indices].to(device, dtype=torch.int64),
            self.next_state[indices].to(device),
            self.reward[indices].to(device),
            indices,
            weights.to(device),
        )

    def update(self, indices, tdiff):
        self.error[indices] = tdiff
        self.max_err = self.error.max()
