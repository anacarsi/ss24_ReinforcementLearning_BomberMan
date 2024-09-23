from typing import List
import torch
import torch.optim as optim
import random
import events as e
from .callbacks import state_to_features, ForthAgentModel,apply_mutations_to_action,device
import settings as s  # use to dynamiclly change the scenario
from datetime import datetime, timedelta
from numpy import ndarray
import numpy as np
from logging import Logger, INFO, DEBUG

from .potential import potential_function
BATCH_SIZE = 3500  # 2-3k
#EPISODES = 100_000  # 30k
from settings import N_ROUNDS
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 500_000  # keep only 1M last transitions
REQUIRE_HISTORY = 450_000  # 100k
RECORD_ENEMY_TRANSITIONS = 0.5  # record enemy transitions with probability ...
from .history import History,HistoryPrioritized
LOG_LEVEL = INFO

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
from settings import SCENARIOS
# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DummySelf:
    transitions: History | HistoryPrioritized
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
    target_network: ForthAgentModel
    trained: int
    score: list[int]


def end_training(self:DummySelf):
    torch.save(self.model.state_dict(), f"model.pth")

    np.save(f"{self.started_training}_loss_history.npy", np.array(self.loss))
    np.save(f"{self.started_training}_score_history.npy", np.array(self.score))
    self.logger.info("Ended training, saved all")

def setup_training(self: DummySelf):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.logger.info(f"Device: {device}")
    self.last_saved = datetime.now() 
    self.started_training = self.last_saved.strftime('%Y.%m.%d-%H.%M.%S')
    self.amount_saved = 0
    #self.transitions = History(TRANSITION_HISTORY_SIZE, self.logger)
    self.transitions = HistoryPrioritized(TRANSITION_HISTORY_SIZE, self.logger)
    
    #self.model.train()  # check what this even does?
    self.epsilon = 0.05
    self.gamma = 0.97
    self.loss_method = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-6)  # 00001 # 1e-5 for later

    self.loss = []
    self.score = []
    self.target_network = ForthAgentModel()
    self.target_network.to(device)
    self.target_network.load_state_dict(self.model.state_dict())
    self.trained = 0

    self.logger.setLevel(LOG_LEVEL)


def train(self: DummySelf):
    self.trained = self.trained + 1
    minibatch = self.transitions.get_batch(BATCH_SIZE)
    if isinstance(minibatch, int):
        if self.trained != 1:
            raise Exception("the batch sampling failed despite working earlier. BUG!")
        self.trained = 0
        self.logger.info(f"not enough experience. The buffer len was: {minibatch} vs {REQUIRE_HISTORY} required")
        return 
    if self.trained % 200 == 0:
        np.save(f"{self.started_training}_loss_history.npy", np.array(self.loss))
        np.save(f"{self.started_training}_score_history.npy", np.array(self.score))
        self.target_network.load_state_dict(self.model.state_dict())
        self.logger.debug("200 train")
        self.logger.info(f"trained at: {self.trained}, {self.transitions.wrapped} {self.transitions.index} epsilon down to: {self.epsilon} lr at {self.optimizer.param_groups[0]["lr"]}")

    self.optimizer.zero_grad()
    state, action, next_state, reward,indices,weights = minibatch

    #yi_arr = torch.zeros(BATCH_SIZE,device=device)  # store the mse.
    #q_value = torch.zeros(BATCH_SIZE,device=device)
    q_value: torch.Tensor = self.model.forward(state)
    q_value = q_value.gather(1, action.unsqueeze(1)).squeeze()
    with torch.no_grad():
        target_q: torch.Tensor = self.target_network.forward(next_state)
        target_q = target_q.max(dim=1).values
    yi_arr = reward + self.gamma * (target_q).nan_to_num(0.0) 
    
    #loss = self.loss_method(q_value, yi_arr) # weights missing
    abs_td_err = (q_value-yi_arr).abs() # needed for history update
    abs_td_err_copy = abs_td_err.detach().clone()  # dont use squared loss, just absolute difference.
    loss = (abs_td_err.pow(2) * weights).mean()
    loss.backward()
    self.optimizer.step()
    self.loss.append(loss.item())
    self.transitions.update(indices,abs_td_err_copy)
    #self.logger.info(f"Loss is: {self.loss[-1]}")

def game_events_occurred(
    self: DummySelf, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py

    old_potential = potential_function(old_game_state, False)
    new_potential = potential_function(new_game_state, False)
    add_reward = self.gamma * new_potential - old_potential
    old_transformed_state,x_flip,y_flip,transpose = state_to_features(old_game_state)
    new_action = ACTIONS.index(apply_mutations_to_action(x_flip,y_flip,transpose, self_action))


    # xp,yp = old_game_state["self"][3]
    # for agent in old_game_state["others"]:
    #     x,y = agent[3]
    #     dist = abs(x - xp) + abs(y - yp)
    #     if dist < 5:
    #         if e.BOMB_DROPPED in events:
    #             add_reward+= 15
    #             break
    # if e.BOMB_EXPLODED in events:
    #     add_reward-=15


    self.transitions.append(
        old_transformed_state,
        new_action,
        state_to_features(new_game_state)[0],
        reward_from_events(self, "forth_agent", events, add_reward),
    )
    #train(self)

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


    end_potential = potential_function(last_game_state, got_killed=(e.GOT_KILLED in events))
    # Using the event could seem like a violation fo the potential rule, but GOT_KILLED is actually a
    # property of the state, so this is fine.
    last_state , x_flip,y_flip,transpose = state_to_features(last_game_state)
    add_reward = -end_potential
    self.transitions.append(
        last_state,
        ACTIONS.index(apply_mutations_to_action(x_flip,y_flip,transpose,last_action)),
        torch.full((5, 17, 17), torch.nan),
        reward_from_events(self, "forth_agent", events, add_reward),
    )
    train(self)
    score = last_game_state["self"][1]
    self.score.append(score)

    self.logger.info(f"Score at end of round:  {score}")

    if self.trained% 1000 == 2:
        now = datetime.now()
        torch.save(
            self.model.state_dict(),
            f"{self.started_training}_snapshot_{now.isoformat()}.pth",
        )
        torch.save(self.model.state_dict(), f"model.pth")
        self.amount_saved += 1
        self.last_saved = now
        self.logger.info("saved snapshot")

        
        self.epsilon = max(0.001,self.epsilon * 0.92)
        for param in self.optimizer.param_groups:
            param["lr"] = param["lr"]* 0.92
    #self.logger.info(f"trained at: {self.trained}, wrapped: {self.transitions.wrapped} ,index: {self.transitions.index} epsilon down to: {self.epsilon} lr for 0th at {self.optimizer.param_groups[0]["lr"]}") 
    # Store the model, to allow reloading it by simultaniously playing agents which are not training
    if last_game_state["round"] % 100 ==47:
        torch.save(self.model.state_dict(),"model.pth")
    if random.random() < 0.001:
        self.logger.info(f"train bei {self.trained}")

def enemy_game_events_occurred(
    self:DummySelf,
    enemy_name: str,
    old_enemy_game_state: dict,
    enemy_action: str,
    enemy_game_state: dict,
    enemy_events: List[str],
):
    if random.random() > RECORD_ENEMY_TRANSITIONS:
        return
    if enemy_action is None:  # clean up the NONE action to waited action
        # this is needed for the Rule based agent
        enemy_action = "WAIT"
        enemy_events.remove("INVALID_ACTION")
        enemy_events.append(e.WAITED)

    old_potential = potential_function(old_enemy_game_state, False)
    new_potential = potential_function(enemy_game_state, False)
    add_reward = self.gamma * new_potential - old_potential
    old_transformed_state,x_flip,y_flip,transpose = state_to_features(old_enemy_game_state)
    
    self.transitions.append(
        old_transformed_state,
        ACTIONS.index(apply_mutations_to_action(x_flip,y_flip,transpose, enemy_action)),
        state_to_features(enemy_game_state)[0],
        reward_from_events(self, enemy_name, enemy_events, add_reward),
    )
    

def reward_from_events(self, agent_name, events: List[str], add=0) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {e.COIN_COLLECTED: 10, e.KILLED_OPPONENT: 50, e.INVALID_ACTION: -10}  # official  # official

    if e.KILLED_OPPONENT in events:
        self.logger.info("Got a kill!!")
    reward_sum = add  # add custo rewards
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(
        f"Awarded to {agent_name}: {reward_sum} for events {', '.join(events)} with potential part: {add}"
    )
    return reward_sum
