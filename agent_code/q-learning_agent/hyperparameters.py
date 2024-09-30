
ROWS = 6
COLS = 6
VISION_RANGE = 3
def training_hyperparameters(self):
    self.epsilon = 0.3
    self.epsilon_decay = 0.9999
    self.epsilon_min = 0.05
    self.learning_rate = 0.3
    self.discount_factor = 0.99
    self.vision_range = VISION_RANGE
    self.rows = ROWS
    self.cols = COLS


def playing_hyperparameters(self):
    #no epsilon-greedy strategy, already explored now only exploitation
    self.epsilon = 0
    self.vision_range = VISION_RANGE
    self.rows = ROWS
    self.cols = COLS