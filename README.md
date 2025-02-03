# Enhanced DQN

This project implements an Enhanced Deep Q-Network (DQN) with Prioritized Experience Replay (PER). The code leverages Gym for the environment, TensorFlow/Keras for deep learning, and a custom SumTree data structure to efficiently sample experiences based on their priority during training.

## Features

- **Deep Q-Learning Algorithm:** Implements DQN with a neural network to approximate Q-values.
- **Prioritized Experience Replay:** Uses a SumTree data structure for efficient sampling of experiences.
- **Gym Integration:** Supports Gym environments for training and evaluation.
- **TensorFlow 2.x & Keras:** Neural network model implemented using TensorFlow and the Keras API.

## Project Structure

- `exp.py`: Main source file which contains the core implementation and training loop.
  - **Lines 1-36:** Imports, definitions, and SumTree class implementation.
  - Remaining sections include building the DQN model, training the agent, and handling environment interactions.

## Prerequisites

- Python 3.12 (or higher)
- [TensorFlow](https://www.tensorflow.org/install)
- [Gym](https://gym.openai.com/) with appropriate dependencies (e.g., Box2D for specific environments)
- Additional libraries:
  - numpy
  - pandas
  - scipy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/enhanced-dqn.git