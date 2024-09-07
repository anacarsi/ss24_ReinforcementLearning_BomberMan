def training_hyperparameters(self):
    # Start with high exploration rate 
    self.epsilon = 0.9
    self.epsilon_decay = 0.9999
    # Minimum exploration rate
    self.epsilon_min = 0.001
    # Low desviacion tipica (ie, learns slower, so each round matters less)
    self.learning_rate = 0.9
    # How much q-values of future stages matter, ie, it determines how quickly 
    # we want to shift the current q-values to the expected q-values
    self.discount_factor = 0.99

def playing_hyperparameters(self):
    #no epsilon-greedy strategy, already explored now only exploitation
    self.epsilon = 0
    self.epsilon_decay = 0.9999
    self.epsilon_min = 0
    # Low desviacion tipica (ie, learns slower, so each round matters less
    self.learning_rate = 0.001 
    # How much q-values of new stage matter 
    self.discount_factor = 0.99