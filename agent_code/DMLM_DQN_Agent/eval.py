from typing import List
import torch


import events as e
import settings as s  # use to dynamiclly change the scenario
from datetime import datetime
from logging import Logger, INFO, DEBUG
from .potential import potential_function
BATCH_SIZE = 1200  # 2-3k
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 500_000  # keep only 1M last transitions
REQUIRE_HISTORY = 450_000  # 100k
RECORD_ENEMY_TRANSITIONS = 1  # record enemy transitions with probability ...
LOG_LEVEL = INFO
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DummySelf:
    logger: Logger
    epsilon: float
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_method: torch.nn.modules.loss._Loss
    started_training: str
    last_saved: datetime
    amount_saved:int
    loss: list[float]
    gamma: float
    trained: int
    score: list[int]


def setup_training(self: DummySelf):
    self.logger.setLevel(LOG_LEVEL)
    self.end_scores = [[],[],[],[]]
    

def game_events_occurred(
    self: DummySelf, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]
):
  print("test")

def end_of_round(self: DummySelf, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    score = last_game_state["self"][1]
    for other_agent in last_game_state["others"]:

        pass

def enemy_game_events_occurred(
    self:DummySelf,
    enemy_name: str,
    old_enemy_game_state: dict,
    enemy_action: str,
    enemy_game_state: dict,
    enemy_events: List[str],
):
    pass
    