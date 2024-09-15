from typing import List
import torch
import random
import events as e
from .callbacks import MODEL_MAP,mutate_point, state_to_features, FifthAgentModel,apply_mutations_to_action,DummySelf, device
import settings as s  # use to dynamiclly change the scenario
from datetime import datetime, timedelta
from numpy import ndarray
import numpy as np
from logging import Logger, INFO, DEBUG
from pathlib import Path
from .potential import potential_function
from .history import History,TRANSITION_HISTORY_SIZE,REQUIRE_HISTORY
BATCH_SIZE = 1  # 2-3k
INIT_LR = 1e-5
#EPISODES = 100_000  # 30k
from settings import N_ROUNDS
# Hyper parameters -- DO modify


RECORD_ENEMY_TRANSITIONS = 0.5  # record enemy transitions with probability ...

LOG_LEVEL = INFO

ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]

# Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))





# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def end_training(self:DummySelf):
    for index, model in enumerate(self.models):
            torch.save(model.state_dict(),f"./models/model{index}.pth")

    #torch.save(self.model.state_dict(), f"/home/david/dev/bomberman_rl/model.pth")

    np.save(f"./{self.started_training}/loss_history.npy", np.array(self.loss))
    np.save(f"./{self.started_training}/score_history.npy", np.array(self.score))
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
    self.last_snap = datetime.now()
    self.started_training = self.last_saved.strftime('%Y.%m.%d-%H.%M.%S')
    self.amount_saved = 0
    self.transitions = History(TRANSITION_HISTORY_SIZE, self.logger)
    self.epsilon = 0.1
    self.gamma = 0.97
    self.loss_method = torch.nn.MSELoss()
    self.optimizers = [torch.optim.Adam(model.parameters(), INIT_LR) for model in self.models]  # 00001 # 1e-5 for later
    self.loss = [list() for _ in range(26)]
    self.score = []
    self.target_networks = [FifthAgentModel() for _ in range(26)]
    for index,model in enumerate(self.target_networks):
        model.to(device,dtype=torch.float32)
        model.load_state_dict(self.models[index].state_dict())
    self.trained = 0
    # for bomb dropping and collection coins:
    # s.SCENARIOS["loot-crate"]["CRATE_DENSITY"] = 0.7
    s.SCENARIOS["loot-crate"]["COIN_COUNT"] = 70
    s.MAX_STEPS = 130
    self.logger.setLevel(LOG_LEVEL)
    path = Path(f"./{self.started_training}/loss")
    path.mkdir(parents=True)

def train(self: DummySelf):
    for x in range(4):
        self.trained = self.trained + 1
        minibatch = self.transitions.get_one()
        if isinstance(minibatch, int):
            if self.trained != 1:
                raise Exception("the sampling failed despite working earlier. BUG!")
            self.trained = 0
            self.logger.info(f"not enough experience. The buffer len was: {minibatch} vs {REQUIRE_HISTORY} required")
            return 
        state, action, next_state, reward ,models= minibatch

        #yi_arr = torch.zeros(BATCH_SIZE,device=device)  # store the mse.
        #q_value = torch.zeros(BATCH_SIZE,device=device)
        first_model_index,second_model_index = models
        self.optimizers[first_model_index].zero_grad()
        q_value: torch.Tensor = self.models[first_model_index].forward(state)
        q_value = q_value[action]
        if second_model_index != -1 :
            with torch.no_grad():
                target_q: torch.Tensor = self.target_networks[second_model_index].forward(next_state)
                #target_q_mask: torch.Tensor = ~target_q[:, 0].isnan()
                target_q = target_q.max(dim=0).values
        else:
            target_q = 0
        
        #yi_arr = reward + self.gamma * (target_q_mask * target_q).nan_to_num(0.0)
        yi_arr = reward + self.gamma * target_q
        loss = self.loss_method(q_value, yi_arr)
        loss.backward()
        self.loss[first_model_index].append(loss.item())
        if random.random() < 0.01:
            self.target_networks[first_model_index].load_state_dict(self.models[first_model_index].state_dict())
            self.target_networks[first_model_index].to(device,dtype=torch.float32)

        self.optimizers[first_model_index].step()
        
    

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
    self.transitions.append(
        old_transformed_state,
        new_action,
        state_to_features(new_game_state)[0],
        reward_from_events(self, "fifth_agent", events, add_reward),
        (MODEL_MAP [mutate_point(old_game_state["self"][3])],MODEL_MAP[mutate_point(new_game_state["self"][3])])
    )
    if random.random() < 0.001:
        self.logger.info(f"train bei {self.trained}")
    train(self)

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
    round = last_game_state["round"]
    # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    end_potential = potential_function(last_game_state, got_killed=(e.GOT_KILLED in events))
    # Using the event could seem like a violation fo the potential rule, but GOT_KILLED is actually a
    # property of the state, so this is fine.
    last_state , x_flip,y_flip,transpose = state_to_features(last_game_state)
    add_reward = - end_potential
    self.transitions.append(
        last_state,
        ACTIONS.index(apply_mutations_to_action(x_flip,y_flip,transpose,last_action)),
        torch.empty((880,)),
        reward_from_events(self, "fifth_agent", events, add_reward),
        (MODEL_MAP [mutate_point(last_game_state["self"][3])],-1)
    )
    train(self)
    score = last_game_state["self"][1]
    self.score.append(score)

    self.logger.info(f"Score at end of round:  {score}")

    now = datetime.now()
    if now - self.last_saved > timedelta(minutes=5):           
        self.epsilon = max(self.epsilon* 0.95,0.01)
        #s.MAX_STEPS = min(400, s.MAX_STEPS + 10 )
        ##s.SCENARIOS["loot-crate"]["CRATE_DENSITY"] = 0.75 +
        # s.SCENARIOS["loot-crate"]["COIN_AMOUNT"] =
        
        self.logger.debug("5min train")
        self.logger.info(f"trained at: {self.trained}, wrapped: {self.transitions.wrapped} ,index: {self.transitions.index} epsilon down to: {self.epsilon} lr for 0th at {self.optimizers[0].param_groups[0]["lr"]}")
        for index,list in enumerate(self.loss):
            np.save(f"./{self.started_training}/loss/loss_history_{index}.npy", np.array(list))
        np.save(f"./{self.started_training}/score_history.npy", np.array(self.score))

        for index,model in enumerate(self.models):
            torch.save(model.state_dict(), f"./models/model{index}.pth")
        self.amount_saved += 1
        self.last_saved = now
        self.logger.info("saved snapshot")
    if now - self.last_snap > timedelta(minutes=30):
        for optimizer in self.optimizers:
           for params in optimizer.param_groups:
                params["lr"] = params["lr"]* 0.5
        #s.MAX_STEPS = min(400, s.MAX_STEPS + 33) 
        self.last_snap = now
        
        for index,model in enumerate(self.models):
            torch.save(
                model.state_dict(),
                f"./{self.started_training}/snapshot_{now.isoformat()}_model_{index}.pth",
            )
    # Store the model

    if last_game_state["round"] == N_ROUNDS:  # it is the last round
        now = datetime.now().isoformat()

        torch.save(self.model.state_dict(), f"{self.started_training}_finished_train_snapshot_model_{now}.pth")
        torch.save(self.model.state_dict(), f"model.pth")

        np.save(f"loss_history_{now}.npy", np.array(self.loss))
        np.save(f"score_history_{now}.npy", np.array(self.score))
        self.logger.info("training finished, saving finished")


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
        (MODEL_MAP [mutate_point(old_enemy_game_state["self"][3])],MODEL_MAP[mutate_point(enemy_game_state["self"][3])])
    )


def reward_from_events(self, agent_name, events: List[str], add=0) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {e.COIN_COLLECTED: 10, e.KILLED_OPPONENT: 50, e.INVALID_ACTION: -50}  # official  # official
    reward_sum = add  # add custo rewards
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.debug(
        f"Awarded to {agent_name}: {reward_sum} for events {', '.join(events)} with potential part: {add}"
    )
    return reward_sum