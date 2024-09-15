import os
import random
import numpy as np
import torch
import logging
from .statics import ACTIONS, ACTIVE_INDICES, MODEL_MAP, DummySelf
from .model import FifthAgentModel
def mutate_point(xy):
    x,y = xy
    if x > 8:
        x = 16 - x
    if y > 8:
        y = 16 - y
    if y > x:
        y, x = x, y
    return (x, y)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
def setup(self: DummySelf):
    """
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("./models/model0.pth"):
        raise Exception("no model found. please look at 'creation.ipnb' in this folder to initialize the model or import them")
    else:
        self.logger.info("Loading model from saved state.")
        # with open("my-saved-model.pt", "rb") as file:
        #     # self.model = pickle.load(file)
        #     self.model = torch.load(file)
        self.logger.setLevel(logging.DEBUG)  # could get overwritten in train.py
        self.models = [FifthAgentModel() for _ in range(26)]
        for index, model in enumerate(self.models):
            model.load_state_dict(
                torch.load(f"./models/model{index}.pth", map_location=device, weights_only=True)
            )
            model.to(device, dtype=torch.float32)
        self.epsilon = 0.4  # gets only used with training and gets overwritten in training code anyway


def act(self: DummySelf, game_state: dict) -> str:
    """
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    
    # with probability epsilon: do random action
    if self.train and random.random() <= self.epsilon:
        return np.random.choice(ACTIONS, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])

    features, x_flip, y_flip, transpose = state_to_features(game_state)
    transformed_actions = [apply_mutations_to_action(x_flip, y_flip, transpose, action) for action in ACTIONS]

    xy = game_state["self"][3]
    xy = mutate_point(xy)

    with torch.no_grad():
        model_out = self.models[MODEL_MAP[xy]].forward(features).flatten().cpu()

    q_vals = {x: y for x, y in zip(transformed_actions, model_out)}
    self.logger.debug(f"The actions were evaluated as follows: {q_vals.items()}")
    probabilites = torch.softmax(model_out * 2, 0).numpy()  # add temperature to the mix (0.2)
    model_choice = np.random.choice(transformed_actions, p=probabilites)
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
    :return: torch.tensor
    """
    if game_state is None:
        return None
    x, y = game_state["self"][3]
    transpose = False
    x_flip = False
    y_flip = False

    flip_dims = []
    result = np.zeros((880), dtype=np.float32)
    bombs = np.zeros((17, 17), dtype=np.float32)
    for xybomb in game_state["bombs"]:
        # tuple unpacking
        xy, bomb_timer = xybomb
        bombs[xy] = bomb_timer + 1.0

    coins = np.zeros((17, 17), dtype=np.float32)
    for coin_cords in game_state["coins"]:
        coins[coin_cords] = 1.0

    agents = np.zeros((17, 17), dtype=np.float32)
    agents[game_state["self"][3]] = 4 + 1 * game_state["self"][2]  # actually use the value

    # it does not matter if the agents are in different orders for each step,
    # as we do not distinguish between them.
    for agent in game_state["others"]:
        agents[agent[3]] = -4 - (1 * agent[2])
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
        result[0:176] = np.flip(game_state["field"], flip_dims).transpose()[ACTIVE_INDICES]
        result[176:352] = np.flip(game_state["explosion_map"], flip_dims).transpose()[ACTIVE_INDICES]
        result[352:528] = np.flip(bombs, flip_dims).transpose()[ACTIVE_INDICES]
        result[528:704] = np.flip(coins, flip_dims).transpose()[ACTIVE_INDICES]
        result[704:880] = np.flip(agents, flip_dims).transpose()[ACTIVE_INDICES]
    else:
        result[0:176] = np.flip(game_state["field"], flip_dims)[ACTIVE_INDICES]
        result[176:352] = np.flip(game_state["explosion_map"], flip_dims)[ACTIVE_INDICES]
        result[352:528] = np.flip(bombs, flip_dims)[ACTIVE_INDICES]
        result[528:704] = np.flip(coins, flip_dims)[ACTIVE_INDICES]
        result[704:880] = np.flip(agents, flip_dims)[ACTIVE_INDICES]

    stacked_channels = torch.from_numpy(result).to(dtype=torch.float32, device=device)
    return (stacked_channels, x_flip, y_flip, transpose)
