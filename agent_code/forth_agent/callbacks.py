import os
import pickle
import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import logging

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class ForthAgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5,32,9,1,3) # 15 x 15 x 32
        self.linear1 = nn.Linear(7200,7200)
        self.linear2 = nn.Linear(7200,1024)
        self.linear3= nn.Linear(1024,6)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.reshape(-1,7200)
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
    if not os.path.isfile("./model.pth"):
        raise Exception("no model created, please look at 'creation.ipnb' in this folder to initialize the model")

    else:
        self.logger.info("Loading model from saved state.")
        # with open("my-saved-model.pt", "rb") as file:
        #     # self.model = pickle.load(file)
        #     self.model = torch.load(file)
        self.logger.setLevel(logging.DEBUG) # could get overwritten in train.py
        self.model = ForthAgentModel()
        self.model.load_state_dict(torch.load("./model.pth",map_location=device))
        self.model.to(device, dtype=torch.float32)
        self.epsilon = 0.4  # gets overwritten in training code anyway

def act(self, game_state: dict) -> str:

    # TODO: only do when other forth agent is training
    if game_state["round"] % 100 == 50 and game_state["step"]==1:
        self.model.load_state_dict(torch.load("./model.pth",map_location=device))

    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    if self.train and random.random() <= self.epsilon:
        return np.random.choice(ACTIONS,p=[0.2,0.2,0.2,0.2,0.1,0.1])

    # training is disabled or exploitation was rolled
    # self.logger.debug("Querying model for action.")
    # features, xflip, yflip = state_to_features(game_state)

    features,x_flip,y_flip,transpose = state_to_features(game_state)
    # ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
    actions = [apply_mutations_to_action(x_flip,y_flip,transpose,action) for action in ACTIONS]

    # dont use the softmax, as the model does not return probabilities, but rather q-values.
    # beacause of this, I don't think we can just interpret them as probabilities and expect a good convergence.
    # model_out = self.model.forward(features).detach().cpu().flatten().softmax(0, torch.float32).numpy()
    with torch.no_grad():
        model_out = self.model.forward(features).flatten().cpu()    
    

    q_vals = {x:y for x,y in zip(actions,model_out)}
    self.logger.debug(f"The actions were evaluated as follows: {q_vals.items()}")
    probabilites = torch.softmax(model_out * 2,0).numpy() # add temperature to the mix (0.2)
    model_choice = np.random.choice(actions,p=probabilites) 
    #model_choice = ACTIONS[torch.argmax(model_out)]

    self.logger.debug(f"ACTION CHOSEN: {model_choice}")
    return model_choice 

def apply_mutations_to_action(x_flip,y_flip,transpose,model_choice):
    x_flip_mapping = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    y_flip_mapping = {"UP": "DOWN", "DOWN": "UP"}
    transpose_mapping = {
        "UP": "LEFT", "LEFT": "UP",
        "DOWN": "RIGHT", "RIGHT": "DOWN"
    }
    if transpose and model_choice in transpose_mapping:
        model_choice = transpose_mapping[model_choice]
    
        
    if x_flip and model_choice in x_flip_mapping:
        model_choice = x_flip_mapping[model_choice]

    if y_flip and model_choice in y_flip_mapping:
        model_choice = y_flip_mapping[model_choice]
    return model_choice



def state_to_features(game_state: dict | None = None) -> tuple[np.array,bool,bool,bool]:
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
    transpose=False
    x_flip = False
    y_flip = False

    flip_dims = []
    result = np.zeros((5,17,17),dtype=np.float32)
    
    for xybomb in game_state["bombs"]:
        # tuple unpacking
        xy, bomb_timer = xybomb
        result[2][xy] = bomb_timer + 1.

    #coins = np.zeros((17, 17),dtype=np.float32)
    for coin_cords in game_state["coins"]:
        result[3][coin_cords] = 1.

    result[4][game_state["self"][3]] = 4 + 1 * game_state["self"][2] # actually use the value

    # it does not matter if the agents are in different orders for each step,
    # as we do not distinguish between them.
    for agent in game_state["others"]:
        result[4][agent[3]] = -4   - (1* agent[2])
    #now apply flips:
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
        result[0]=np.flip(game_state["field"],flip_dims).transpose()
        result[1]=np.flip(game_state["explosion_map"],flip_dims).transpose()
        result[2]=np.flip(result[2],flip_dims).transpose()
        result[3]=np.flip(result[3],flip_dims).transpose()
        result[4]=np.flip(result[4],flip_dims).transpose()
    else:
        result[0]=np.flip(game_state["field"],flip_dims)
        result[1]=np.flip(game_state["explosion_map"],flip_dims)
        result[2]=np.flip(result[2],flip_dims)
        result[3]=np.flip(result[3],flip_dims)
        result[4]=np.flip(result[4],flip_dims)

    stacked_channels = torch.from_numpy(result).to(dtype=torch.float32, device=device)
    return (stacked_channels,x_flip,y_flip,transpose)
