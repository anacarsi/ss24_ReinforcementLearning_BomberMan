import os
import pickle
import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import logging

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

INVERSE_TEMPERATURE =50
#INVERSE_TEMPERATURE_1=20
#! check if cpu or cuda is selected
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
OTHER_TRAIN = False


class ForthAgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 32, 9, 1, 3)  # 15 x 15 x 32
        self.linear1 = nn.Linear(7200, 7200)
        self.linear2 = nn.Linear(7200, 1024)
        self.linear3 = nn.Linear(1024, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.reshape(-1, 7200)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #train:
    if self.train:
        if not os.path.isfile("./model.pth"):
            raise Exception("no model created, please look at 'creation.ipnb' in this folder to initialize the model")
        self.model = ForthAgentModel()
        self.model.load_state_dict(torch.load("./model.pth", map_location=device))
        self.model.to(device, dtype=torch.float32)
    else:
        # inference:
        if not os.path.isfile("./model_1.pth") or not os.path.isfile("./model_4.pth"):
            raise Exception("no model created, please look at 'creation.ipnb' in this folder to initialize the model")
        self.model = ForthAgentModel()
        self.model.load_state_dict(torch.load("./model_4.pth", map_location=device))
        self.model.to(device, dtype=torch.float32)
        self.model_1= ForthAgentModel() # model 1 is used when only one agent is remaining
        self.model_1.load_state_dict(torch.load("./model_1.pth",map_location=device))
        self.model_1.to(device,dtype=torch.float32)


    self.logger.info("Loading model from saved state.")
    self.epsilon = 0.0  # gets overwritten in training code anyway
    self.logger.setLevel(logging.INFO)  # could get overwritten in train.py

    

def act(self, game_state: dict) -> str:

    # # only do when other forth agent is training
    # if OTHER_TRAIN==True:
    #     if game_state["round"] % 100 == 50 and game_state["step"] == 1:
    #         self.model.load_state_dict(torch.load("./model.pth", map_location=device))

    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train and random.random() < self.epsilon:
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    features, x_flip, y_flip, transpose = state_to_features(game_state)
    # reminder: ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
    actions = [apply_mutations_to_action(x_flip, y_flip, transpose, action) for action in ACTIONS]
    with torch.no_grad():
        if self.train:
            model_out = self.model.forward(features).flatten().cpu()
        else:
            if len(game_state["others"]) < 2:
                self.logger.info("used 1")
                model_out = self.model_1.forward(features).flatten().cpu()
            else:
                self.logger.info("used 4")
                model_out = self.model.forward(features).flatten().cpu()

    q_vals = {x: y for x, y in zip(actions, model_out)}
    self.logger.debug(f"The actions were evaluated as follows: {q_vals.items()}")
    probabilites = torch.softmax(
        model_out * INVERSE_TEMPERATURE, 0
    ).numpy()  # add temperature to the mix (0.5 => 2 as a factor)
    #(This increases the probabilities of being picked of the actions with higher expected reward,
    # while the worse actions will get played even less)
    model_choice = np.random.choice(actions, p=probabilites)
    #model_choice = actions[torch.argmax(model_out)]

    self.logger.debug(f"ACTION CHOSEN: {model_choice}")
    return model_choice


def apply_mutations_to_action(x_flip, y_flip, transpose, model_choice):
    x_flip_mapping = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    y_flip_mapping = {"UP": "DOWN", "DOWN": "UP"}
    transpose_mapping = {"UP": "LEFT", "LEFT": "UP", "DOWN": "RIGHT", "RIGHT": "DOWN"}
    if transpose and model_choice in transpose_mapping:
        model_choice = transpose_mapping[model_choice]

    if x_flip and model_choice in x_flip_mapping:
        model_choice = x_flip_mapping[model_choice]

    if y_flip and model_choice in y_flip_mapping:
        model_choice = y_flip_mapping[model_choice]
    return model_choice


def state_to_features(game_state: dict | None = None) -> tuple[np.array, bool, bool, bool]:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.

    :return: np.array (length: 176 *2 + 12 + 8 = 372)
    """
    if game_state is None:
        return None
    x, y = game_state["self"][3]
    transpose = False
    x_flip = False
    y_flip = False

    flip_dims = []
    result = np.zeros((5, 17, 17), dtype=np.float32)

    for xybomb in game_state["bombs"]:
        # tuple unpacking
        xy, bomb_timer = xybomb
        result[2][xy] = bomb_timer + 1.0

    # coins = np.zeros((17, 17),dtype=np.float32)
    for coin_cords in game_state["coins"]:
        result[3][coin_cords] = 1.0

    result[4][game_state["self"][3]] = 4 + 1 * game_state["self"][2]  # actually use the value

    # it does not matter if the agents are in different orders for each step,
    # as we do not distinguish between them.
    for agent in game_state["others"]:
        result[4][agent[3]] = -4 - (1 * agent[2])
    # now apply flips:
    if x > 8:
        flip_dims.append(0)
        x_flip = True
        x = 16 - x
    if y > 8:
        flip_dims.append(1)
        y_flip = True
        y = 16 - y
    if y > x:
        transpose = True
        result[0] = np.flip(game_state["field"], flip_dims).transpose()
        result[1] = np.flip(game_state["explosion_map"], flip_dims).transpose()
        result[2] = np.flip(result[2], flip_dims).transpose()
        result[3] = np.flip(result[3], flip_dims).transpose()
        result[4] = np.flip(result[4], flip_dims).transpose()
    else:
        result[0] = np.flip(game_state["field"], flip_dims)
        result[1] = np.flip(game_state["explosion_map"], flip_dims)
        result[2] = np.flip(result[2], flip_dims)
        result[3] = np.flip(result[3], flip_dims)
        result[4] = np.flip(result[4], flip_dims)

    stacked_channels = torch.from_numpy(result).to(dtype=torch.float32, device=device)
    return (stacked_channels, x_flip, y_flip, transpose)
