import os
import pickle
import random

import numpy as np

import torch
from torch import nn
import logging
active_indices = np.array(
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
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class FirstAgentModel(nn.Module):
    active_indices: np.ndarray[int]

    def __init__(self, input_size: int, output_size: int = 6):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.main( x)


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
    if not os.path.isfile("model.pth"):
        raise Exception("no model created, please look at 'creation.ipnb' in this folder to initialize the model")

    else:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.logger.info("Loading model from saved state.")
        # with open("my-saved-model.pt", "rb") as file:
        #     # self.model = pickle.load(file)
        #     self.model = torch.load(file)
        self.model = FirstAgentModel(372)
        self.model.load_state_dict(torch.load("model.pth"))
        self.model.to(device,dtype=torch.float32)
        self.epsilon = 0.0 # gets overwritten in training code


def apply_transmutations_tuples(a: tuple[int, int], xflip, yflip) -> tuple[int, int]:
    if xflip:
        a = (16 - a[0], a[1])
    if yflip:
        a = (a[0], 16 - a[1])
    return a


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    # epsilon = 0.9 # with prop epsilon, a random action is chosen (exploration)
    # epsilon will be set in setup
    if self.train and random.random() <= self.epsilon:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    # training is disabled or exploitation was rolled
    #self.logger.debug("Querying model for action.")
    features, xflip, yflip = state_to_features(game_state)
    model_out = self.model.forward(features).softmax(0,torch.float32).detach().cpu().numpy()
    model_choice = np.random.choice(ACTIONS, p=model_out )
    if model_choice == "WAIT" or model_choice== "BOMB":
        return model_choice
    # else:
    
    if xflip:
        if model_choice=="RIGHT":
            return "LEFT"
        if model_choice=="LEFT":
            return "RIGHT"
    
    if yflip:
        if model_choice=="UP":
            return "DOWN"
        if model_choice=="DOWN":
            return "UP"
    #else
    return model_choice # either no mutation or mutation does not change direction


def state_to_features(game_state: dict | None =None) -> np.array:
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

    # mutate the state so that the agent will always have x,y <= 8, (is in the top left corner)
    # TODO: in the future, maybe aditionally ensure that x>=y, to exploit even more symetry.

    x, y = game_state["self"][3]
    # transpose=False # for later
    x_flip = False
    y_flip = False
    # these values get checked later

    
    # 0 means is always a hard wall, 1 means is empty or chest or coin or explosion

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    n1, n2 = game_state["field"].shape
    field_copy: np.ndarray = game_state["field"]  # TODO: check if modifying this is allowed or copy is needed
    for coin_cords in game_state["coins"]:
        field_copy[(coin_cords[1], coin_cords[0])] = 1

    explosion_copy: np.ndarray = game_state["explosion_map"]

    # now apply flips:
    if x > 8:
        x_flip = True
        field_copy = np.flipud(field_copy)
        explosion_copy = np.flipud(explosion_copy)
    if y > 8:
        y_flip = True
        field_copy = np.fliplr(field_copy)
        explosion_copy = np.fliplr(explosion_copy)

    channels = [
        field_copy[active_indices],
        explosion_copy[active_indices],
    ]  # we remove the walls that will always be ther

    #

    # we could also try to place the bombs into the game_state array,
    # but for now we express them as new features

    # add all the bombs (amount 4, features per bomb: 3, x,y,timer)  amounts to 12 features
    # if bomb is missing, set all  to -1
    bombs = np.full((12,), -1)
    for index, xybomb in enumerate(
        game_state["bombs"][0:4]
    ):  # Note: only encode 4 bombs, keep in mind for training to never place more than 4 bombs!
        xy, bomb = xybomb
        xy = apply_transmutations_tuples(xy, x_flip, y_flip)
        # tuple unpacking
        bomb_starting_index = index * 3
        bombs[bomb_starting_index + 1 : bomb_starting_index + 3] = xy
        bombs[bomb_starting_index + 3] = bomb

    channels.append(bombs)

    # now pass position of each agent (we, enemy 1, 2,3)
    agent_positions = np.full((8,), -1)
    # our agent:
    agent_positions[0:2] = apply_transmutations_tuples(game_state["self"][3], xflip=x_flip, yflip=y_flip)

    # it does not matter if the agents are in different orders for each step,
    # as we do not distinguish between them.
    for index, agent in enumerate(game_state["others"]):
        array_agent_start_index = index * 2 + 2  # same as (index + 1) * 2
        agent_positions[array_agent_start_index : array_agent_start_index + 2] = apply_transmutations_tuples(
            agent[3], xflip=x_flip, yflip=y_flip
        )

    channels.append(agent_positions)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels =torch.tensor( np.concatenate(channels,axis=None,dtype=np.float32),device=device,dtype=torch.float32)
    # and return them as a vector
    return (stacked_channels, x_flip, y_flip)
