# 🦖 DINO Game Reinforcement Learning Project

A reinforcement learning project where an AI agent learns to play the Chrome Dinosaur game using Deep Q-Network (DQN) implemented in **PyTorch**.

## 🎯 Project Overview

This project trains a deep reinforcement learning agent to master the Chrome Dinosaur game using a local Pygame replica. The agent learns through trial and error, developing strategies to avoid obstacles (cacti for now, birds later) and maximize its survival time.

### Credits
- **Game Engine**: [Chrome-Dino-Runner](https://github.com/dhhruv/Chrome-Dino-Runner) by [dhhruv](https://github.com/dhhruv) - A Pygame replica of the Chrome Dino game

## 🛠️ Technical Stack

- **Python**: 3.11 (Conda DINO_TF environment)
- **Deep Learning Framework**: PyTorch 2.5.1 + CUDA 12.1
- **Game Engine**: Pygame (Chrome-Dino-Runner)
- **RL Environment**: Gymnasium (OpenAI Gym replacement)
- **Hardware**: Optimized for i5-12450H + RTX 3050 Mobile (4GB VRAM)

### Key Dependencies
- `pygame` - Game engine for local training
- `gymnasium` - RL environment interface
- `torch` - Deep learning framework (with CUDA)
- `numpy` - Numerical operations

## 📋 Project Structure

```
Dino RL/
├── dino_rl_plan.txt              # Detailed project roadmap
├── PROJECT_CHANGELOG.txt          # Project decisions and changes
├── README.md                      # This file
├── state_based_training.ipynb     # Main training notebook
├── Chrome-Dino-Runner/            # Pygame Dino game (by dhhruv)
│   ├── chromedino.py             # Game source
│   └── assets/                    # Game sprites
└── src/
    ├── dino_pygame_env.py        # Pygame Gym environment wrapper
    ├── dino_state_env.py         # Browser-based environment (legacy)
    └── dino_game.py              # Original game implementation
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
pip install -r requirements.txt
```

4. **Download ChromeDriver**:
   - Check your Chrome version: `chrome://settings/help`
   - Download matching ChromeDriver from https://googlechromelabs.github.io/chrome-for-testing/
   - Place `chromedriver.exe` in your PATH or project folder
   - See [BROWSER_SETUP.md](BROWSER_SETUP.md) for detailed instructions

## 🎮 Browser Environment

### Quick Test

Test the browser automation environment:

```bash
python src/browser_dino_env.py
```

This will:
- Open Chrome and navigate to https://chromedino.com/
- Run a random agent for 100 steps
- Display FPS and score information

### Using in Python

```python
from src.browser_dino_env import BrowserDinoEnv

# Create environment
env = BrowserDinoEnv(
    headless=False,      # Set True to hide browser
    target_fps=20,       # Target frame rate
    chromedriver_path=None  # Auto-detect from PATH
)

# Standard Gym interface
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated:
        obs, info = env.reset()

env.close()
```

### Environment Details

**Observation Space**:
- 80x80 grayscale screenshot of game area
- Preprocessed from browser capture
- Values: 0-255 (uint8)

**Action Space** (2 discrete actions):
- 0: Do nothing (run)
- 1: Jump (press Space)

**Reward Structure:**
- +0.1 per frame survived
- +0.1 × score_increase for passing obstacles
- -10.0 for game over

**Performance**:
- Target: 20 FPS (configurable)
- Supports 2-3 parallel instances
- Headless mode for faster training

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
