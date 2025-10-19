# 🦖 DINO Game Reinforcement Learning Project

A reinforcement learning project where an AI agent learns to play the Chrome Dinosaur game using Deep Q-Network (DQN) implemented in TensorFlow.

## 🎯 Project Overview

This project trains a deep reinforcement learning agent to master the classic Chrome Dinosaur game. The agent learns through trial and error, developing strategies to avoid obstacles and maximize its survival time.

## 🛠️ Technical Stack

- **Python**: 3.14
- **Deep Learning Framework**: TensorFlow
- **Game Engine**: Pygame
- **RL Environment**: OpenAI Gym
- **Environment Manager**: Conda (environment: DINO)

## 📋 Project Structure

```
Dino RL/
├── dino_rl_plan.txt           # Detailed project roadmap
├── PROJECT_CHANGELOG.txt       # Project decisions and changes
├── README.md                   # This file
├── model.ipynb                 # Main model implementation (Coming soon)
├── requirements.txt            # Python dependencies (Coming soon)
└── src/                        # Source code (Coming soon)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.14
- Conda package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SaiGuruInukurthi/DINO-game-RL-trial-.git
cd DINO-game-RL-trial-
```

2. Activate the conda environment:
```bash
conda activate DINO
```

3. Install dependencies:
```bash
pip install pygame gym tensorflow numpy matplotlib
```

## 🎮 How It Works

### Game Environment
- The agent interacts with a Pygame-based dinosaur game
- Observations include obstacle positions, distances, and game state
- Actions: Jump, Duck, or Do Nothing

### RL Algorithm
- **Algorithm**: Deep Q-Network (DQN)
- **Neural Network**: TensorFlow-based model
- **Training**: Experience replay and epsilon-greedy exploration

### Reward System
- +0.1 points for each frame survived
- +10 points for passing obstacles
- -100 points for collisions

## 📊 Training Process

The model is trained through:
1. **Exploration**: Agent tries random actions
2. **Experience Collection**: Storing game states and outcomes
3. **Learning**: Neural network learns optimal actions
4. **Exploitation**: Agent uses learned strategies

## 🎯 Project Goals

- ✅ Set up development environment
- 🔄 Implement game environment with Gym interface
- 🔄 Build DQN agent with TensorFlow
- 🔄 Train agent to achieve consistent high scores
- 🔄 Visualize training progress and gameplay

## 📈 Success Metrics

- Agent survives > 1000 game steps
- Achieves consistent scores > 500 points
- Successfully avoids multiple obstacle types
- Training convergence within 2000 episodes

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

## 📝 License

Open source - feel free to use and modify.

## 🔗 Resources

- [Chrome Dino Game](chrome://dino)
- [OpenAI Gym Documentation](https://gymnasium.farama.org/)
- [TensorFlow RL Guide](https://www.tensorflow.org/agents)
- [DQN Paper](https://arxiv.org/abs/1312.5602)

## 👨‍💻 Author

SaiGuruInukurthi

---

**Status**: 🚧 In Development

Last Updated: October 19, 2025
