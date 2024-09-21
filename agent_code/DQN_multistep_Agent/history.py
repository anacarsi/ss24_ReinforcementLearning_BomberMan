from .train import TRANSITION_HISTORY_SIZE,device,REQUIRE_HISTORY,STEPS_FUTURE
import torch
import random

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

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

    def __getitem__(self, idx):
            if isinstance(idx, int):
                return self.state[idx], self.action[idx], self.next_state[idx], self.reward[idx]
            else:
                raise TypeError("Index must be an integer")

    def __setitem__(self, idx, pos, value):
        if isinstance(idx, int):
            if pos == 0:
                self.state[idx] = value
            elif pos == 1:
                self.action[idx] = value
            elif pos == 2:
                self.next_state[idx] = value
            elif pos == 3:
                self.reward[idx] = value        
            else:
                raise ValueError("Position must be an integer between 0 and 3")
        else:
            raise TypeError("Index must be an integer")

    def __iter__(self):
        for i in range(self.index):
            yield self.state[i], self.action[i], self.next_state[i], self.reward[i]
            

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
    
    def nextkrewards_kstate(self, action, next_state, reward, lists: dict) -> tuple | None:
        """
        Calculate the next k rewards and k states based on a given action and next state.
        This function processes the history of transitions to compute the rewards and states 
        for the next k steps (defined by STEPS_FUTURE) given a specific action and next state. 
        It also updates the provided lists dictionary with the current transition if certain 
        conditions are met.
        Args:
            action (int): The action taken.
            next_state (Any): The next state resulting from the action.
            reward (float): The reward received for the action.
            lists (dict): A dictionary containing lists of transitions indexed by actions.
        Returns:
            tuple: A tuple containing:
                - rewards (list): A list of rewards for the next k steps.
                - state (Any): The state after k steps.
        """
        rewards = [reward]
        states = []
        transitions = [self[self.index - 1]]
        for j in range(1, STEPS_FUTURE + 1):
            for tr in self:
                if tr[1] == action and tr[0] == next_state:
                    rewards.append(tr[3])
                    transitions.append(tr)
                    if j == STEPS_FUTURE or j == (STEPS_FUTURE - 1):
                        states.append(tr[0])
                        break
                    next_state = tr[0]
                break
            #cojo del diccionario de listas las que corresponden a mi accion
            #las listas contienen indices adecuados de la parte de cada transicion
            for list in lists[ACTIONS[action]]:
                print("ACTIONS[action] is: " + str(ACTIONS[action]))
                #si el anterior nuevo estado es nuestro estado actual
                if list[j-1][2] == self.state:
                    list[j].append(self)
                    if len(list) == STEPS_FUTURE:
                        rewards = []
                        for i in list:
                            rewards.append(i[3])
                        list.clear()
                        print(f"with completed list: rewards {rewards}, states: {states}")
                        return rewards, (self.state, self.next_state)
        if len(rewards) == STEPS_FUTURE :
            print(f"without list: rewards {rewards}, states: {states}")
            return rewards, states
        else:
            #guardar lista con lo que tenemos hasta ahora, y si es solo un elemento, pues solo 1
            #TODO: esta esto añadiendome una nueva lista a esa lista o añadiendo a la lista que ya existe transitions?
            list[ACTIONS[action]].append(transitions)
            print("ACTIONS[action] is: " + str(ACTIONS[action]))
            print(f"with incomplete list: length list: {len(list)}")
            return None
 
