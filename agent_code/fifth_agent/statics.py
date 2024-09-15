
import numpy as np
from datetime import datetime
from logging import Logger
from .history import History
import torch
from .model import FifthAgentModel 
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTIVE_INDICES = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],dtype=bool
    )

MODEL_MAP = {(1, 1): 0, (2, 1): 1, (3, 1): 2, (4, 1): 3, (5, 1): 4, (6, 1): 5, (7, 1): 6, (8, 1): 7, (3, 2): 8, (5, 2): 9, (7, 2): 10, (3, 3): 11, (4, 3): 12, (5, 3): 13, (6, 3): 14, (7, 3): 15, (8, 3): 16, (5, 4): 17, (7, 4): 18, (5, 5): 19, (6, 5): 20, (7, 5): 21, (8, 5): 22, (7, 6): 23, (7, 7): 24, (8, 7): 25}

class DummySelf:
    train:bool
    transitions: History
    logger: Logger
    epsilon: list[float]
    models: list[torch.nn.Module]
    optimizers: list[torch.optim.Optimizer]
    loss_method: torch.nn.modules.loss._Loss
    started_training: str
    last_saved: datetime
    amount_saved: int
    loss: list[float]
    gamma: float
    target_networks: list[FifthAgentModel]
    trained: int
    score: list[int]