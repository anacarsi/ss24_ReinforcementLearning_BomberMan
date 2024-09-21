import os
from pickle import load as pickle_load
import numpy as np
from . import auxiliary_functions as a
from . import hyperparameters as hyp


ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

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
    self.train=False
    #HYPERPARAMETERS
    if os.path.isfile("q_table.pickle"):
        with open("q_table.pickle", "rb") as file:
            self.q_table = pickle_load(file)
    else:
        self.q_table = {}

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.
    
    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
   
    #epsilon-greedy strategy
    if np.random.random() > self.epsilon and self.train==False:
        action = np.argmax(self.q_table.get(self.state, [-10] * 6))
    else:
        action = np.random.randint(0, 6)
    self.action = action 
    
    return ACTIONS[action]
    
from typing import NamedTuple

class Features(NamedTuple):
    bx:int# 1: nearest bomb has greater x value
    by:int
    cx:int
    cy:int
    crx:int
    cry:int
    sd:bool
    su:bool
    sl:bool
    sr:bool
    ss:bool
    up:bool
    dp:bool
    rp:bool
    lp:bool

def state_to_features(game_state) -> Features:
    xp,yp = game_state["self"][3]
    if len(game_state["bombs"]) ==0:
        bx=0,by=0
    else:
        distance = 100 # will get overwritten
        current_nearest_bomb_index = 0
        nearest_bomb_index=0
        for bombxy,bombt in game_state["bombs"]:
            x,y = bombxy
            next_distance = min(distance, abs(x - xp) + abs(y - yp))
            if next_distance< distance:
                distance=next_distance
                nearest_bomb_index = current_nearest_bomb_index
            current_nearest_bomb_index+=1
        if xp < game_state["bombs"][nearest_bomb_index][0][0]:
            bx=1
        elif xp ==game_state["bombs"][nearest_bomb_index][0][0]:
            bx = 0
        else:
            bx=-1
        if yp < game_state["bombs"][nearest_bomb_index][0][1]:
            by=1
        elif yp ==game_state["bombs"][nearest_bomb_index][0][1]:
            by = 0
        else:
            by=-1
    

    if len(game_state["coins"]) ==0:
        cx=0,cy=0
    else:
        distance = 100 # will get overwritten
        current_nearest_bomb_index = 0
        nearest_bomb_index=0
        for x,y in game_state["coins"]:
            next_distance = min(distance, abs(x - xp) + abs(y - yp))
            if next_distance< distance:
                distance=next_distance
                nearest_bomb_index = current_nearest_bomb_index
            current_nearest_bomb_index+=1
        if xp < game_state["coins"][nearest_bomb_index][0]:
            cx=1
        elif xp ==game_state["coins"][nearest_bomb_index][0]:
            cx = 0
        else:
            cx=-1
        if yp < game_state["coins"][nearest_bomb_index][1]:
            cy=1
        elif yp ==game_state["coins"][nearest_bomb_index][1]:
            cy = 0
        else:
            cy=-1


    #bomb done: now: coin


def is_safe(game_state, got_killed: bool):
    """
    inputs:
    player_cords: tuple(int,int) being the x and y coords
    bobs_ticking: the list of bombs
    field: the field with only walls, crates and air

    returns:
    bool: True if player is currently not in any blast radius.
    Else false
    """
    if got_killed:
        return True
    player_cords = game_state["self"][3]
    bombs_ticking = game_state["bombs"]
    field = game_state["field"]
    xp, yp = player_cords
    for xy, _timer in bombs_ticking:
        xb, yb = xy
        if xp == xb:
            if yp in range(yb - 3, yb + 4):
                if (yb - yp) == 2:
                    if field[(xp, yb - 1)] != 0:
                        continue
                elif (yp - yb) == 2:
                    if field[(xp, yb + 1)] != 0:
                        continue
                return False
        if yp == yb:
            if xp in range(xb - 3, xb + 3):
                if (xb - xp) == 2:
                    if field[(xp + 1, yb)] != 0:
                        continue
                elif (xp - xb) == 2:
                    if field[(xp - 1, yb)] != 0:
                        continue
                return False
    return True