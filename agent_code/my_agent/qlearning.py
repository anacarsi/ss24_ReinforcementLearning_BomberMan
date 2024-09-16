# model for QLearning
import numpy as np


class QLearningModel():
    def __init__(self):
        self.q_table = {}
        self.epsilon = 1.0  # Exploration rate for epsilon-greedy
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.q_table = {} 
        self.task = 1

    def define_task(self, task) -> None:
        """
        Define actions depending on the task.
        Task 1: Collect coins as quickly as possible.
        Task 2: Collect hidden coins within a step limit.
        Task 3: Hunt and blow up a predefined peaceful agent.
            Task 3.2: Hunt and blow up a coin collector agent.
        Task 4: Survive and collect coins.

        :param task: The task to be performed.
        """
        if task == 1:
            self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
        elif task > 1:
            self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    
    def load_table(self, q_table) -> None:
        """
        Load the Q-table from a file if it exists.

        :param q_table: The Q-table to be loaded.
        """
        self.q_table = q_table
    
    def choose_action(self, state) -> str:
        """
        Choose the next action based on the current state using an epsilon-greedy policy and Q-values.

        :param state: The current state of the game.
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            candidates = self.q_table.get(tuple(state), np.zeros(len(self.actions)))
            actio = self.actions[np.argmax(candidates)]
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return action
    
