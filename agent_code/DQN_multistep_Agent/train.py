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
BATCH_SIZE = 1200  # 2-3k
#EPISODES = 100_000  # 30k
# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 500_000  # keep only 1M last transitions
REQUIRE_HISTORY = 450_000  # 100k
RECORD_ENEMY_TRANSITIONS = 1  # record enemy transitions with probability ...
from .history import History
LOG_LEVEL = INFO
STEPS_FUTURE = 4 #Hyperparameter of multistep learning


ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DummySelf:
    transitions: History
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

    #np.save(f"{self.started_training}_loss_history.npy", np.array(self.loss))
    #np.save(f"{self.started_training}_score_history.npy", np.array(self.score))
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
    self.transitions = History(TRANSITION_HISTORY_SIZE, self.logger)
    self.model.train()  # check what this even does?
    self.epsilon = 0.05
    self.gamma = 0.97
    self.loss_method = torch.nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), 5e-6)  # 00001 # 1e-5 for later

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
    #update target network each 200 episodes
    if self.trained % 200 == 0:
        #np.save(f"{self.started_training}_loss_history.npy", np.array(self.loss))
        #np.save(f"{self.started_training}_score_history.npy", np.array(self.score))
        self.target_network.load_state_dict(self.model.state_dict())
        self.logger.debug("200 train")
        self.logger.info(f"trained at: {self.trained}, {self.transitions.wrapped} {self.transitions.index} epsilon down to: {self.epsilon} lr at {self.optimizer.param_groups[0]["lr"]}")

    self.optimizer.zero_grad()
    state, action, next_state, reward = minibatch

    q_value: torch.Tensor = self.model.forward(state)
    q_value = q_value.gather(1, action.unsqueeze(1)).squeeze()
    with torch.no_grad():
        target_q: torch.Tensor = self.target_network.forward(next_state)
        target_q = target_q.max(dim=1).values
        #compute rewards taking the same action for the next STEPS_FUTURE steps
        rewards = [reward]
        gammas = [self.gamma]
        for i in range(2, STEPS_FUTURE):
            next_state = self.target_network.forward(next_state).max(dim=1).indices
            reward = self.transitions.reward[next_state]
            rewards.append(reward)
            gammas.append(gammas[-1]*self.gamma)
        gammas.append(gammas[-1]*self.gamma)
        #compute truncated n-step return
        reward = sum([r*g for r,g in zip(rewards,gammas)])

    #TD target with multistep learning
    yi_arr = reward + gammas[-1] * (target_q).nan_to_num(0.0)
    loss = self.loss_method(q_value, yi_arr)

    loss.backward()
    self.optimizer.step()
    self.loss.append(loss.item())


def game_events_occurred(
    self: DummySelf, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]
):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # state_to_features is defined in callbacks.py
    old_potential = potential_function(old_game_state, False)
    new_potential = potential_function(new_game_state, False)
    add_reward = self.gamma * new_potential - old_potential
    old_transformed_state,x_flip,y_flip,transpose = state_to_features(old_game_state)
    new_action = ACTIONS.index(apply_mutations_to_action(x_flip,y_flip,transpose, self_action))
    self.transitions.append(
        old_transformed_state,
        new_action,
        state_to_features(new_game_state)[0],
        reward_from_events(self, "forth_agent", events, add_reward),
    )

def end_of_round(self: DummySelf, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

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

    #epsilon and learning rate decay
    if self.trained % 1_000 == 0:
        self.epsilon = max(0.01,self.epsilon * 0.9)
        for param in self.optimizer.param_groups:
            param["lr"] = param["lr"]* 0.95
        self.logger.info(f"trained at: {self.trained}, wrapped: {self.transitions.wrapped} ,index: {self.transitions.index} epsilon down to: {self.epsilon} lr for 0th at {self.optimizer.param_groups[0]["lr"]}") 
    
    #resetting epsilon and learning rate every 20_000 episodes
    if self.trained % 20_000 == 0:
        self.epsilon = 0.05
        for param in self.optimizer.param_groups:
            param["lr"] = 5e-6
       
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
    game_rewards = {e.COIN_COLLECTED: 10, e.KILLED_OPPONENT: 50, e.INVALID_ACTION: -10,
                     (e.COLLECTED_EVERYTHING and e.SURVIVED_ROUND): 20}  # official  # official 
                    #custom #custom: collected everything, survived round and didn't wait until 
                    #the end

    reward_sum = add  # add custo rewards
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(
        f"Awarded to {agent_name}: {reward_sum} for events {', '.join(events)} with potential part: {add}"
    )
    return reward_sum
