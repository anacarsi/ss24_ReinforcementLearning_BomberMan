# ss24_ReinforcementLearning_BomberMan
Repository for the implementation of a reinforcement learning agent for the game of Bomber Man. 
# Bomberman Reinforcement Learning Agent

![Bomberman RL](./images/cover-image.png){:height="300px" width="300px"} 
## Project Description

## Project Description

This project implements two reinforcement learning agents for the classic Bomberman game. The agent is designed to navigate the game board, collect coins, and compete against other agents using various strategies. The goal is to develop an intelligent agent that can adapt to different game scenarios and outperform its opponents.

## Table of Contents

- [Agent Overview](#agent-overview)
- [Callbacks](#callbacks)
- [Command Line Instructions](#command-line-instructions)
- [Requirements](#requirements)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Agent Overview

The agent operates in a discrete step environment where it can perform actions such as moving, dropping bombs, or waiting. The agent learns through reinforcement learning techniques, adapting its strategy based on the game state and feedback from the environment.

### Key Features

- Efficient navigation to collect coins.
- Bomb usage to destroy crates and opponents.
- Adaptability to different game scenarios and opponents.

## Callbacks

The agent interacts with the game environment through a series of callbacks. These callbacks handle the setup, action decisions, and event logging during the game. Key functions include:

- `setup(self)`: Initializes the agent before the first round.
- `act(self, game_state)`: Determines the best action based on the current game state.

### Event Logging

During training, various events are logged to provide feedback for the agent's learning process. Events include actions taken, coins collected, and opponents eliminated.

## Command Line Instructions

To run the project, use the following command line instructions:

```bash
# To play the game
python main.py play

# To run in training mode
python main.py train

# To save a replay of the game
python main.py replay <stored-replay>

# To skip frames for faster execution
python main.py --skip-frames

# To run without GUI for efficient training
python main.py --no-gui

Requirements

    Python 3.7 or higher
    Required libraries specified in requirements.txt

Hardware Requirements

    Processor: AMD Ryzen 5 2600 or equivalent
    RAM: Minimum 8 GB
    GPU: Recommended for deep learning (NVIDIA preferred)

Installation

    Clone the repository:

    git clone https://github.com/anacarsi/ss24_ReinforcementLearning_BomberMan.git
    cd bomberman-rl-agent

    Install the required libraries:

    pip install -r requirements.txt

Usage

After installation, you can start training your agent or playing the game using the command line instructions provided above. Customize your agent's behavior by modifying the callbacks.py file.
Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
License

This project is licensed under the MIT License. See the LICENSE file for details.

