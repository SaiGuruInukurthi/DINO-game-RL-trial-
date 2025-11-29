# 🦖 NEAT Training for Chrome Dino Game

A neuroevolution project where an AI learns to play the Chrome Dinosaur game using **NEAT** (NeuroEvolution of Augmenting Topologies) implemented with `neat-python`.

## 🎯 Project Overview

This project evolves neural networks to master the Chrome Dinosaur game using a local Pygame replica. The networks evolve both their **weights** and **topology** (structure) through genetic algorithms, learning to jump over cacti and duck under birds.

### Credits
- **Game Engine**: Based on [Chrome-Dino-Runner](https://github.com/dhhruv/Chrome-Dino-Runner) by [dhhruv](https://github.com/dhhruv)

## 🧬 NEAT Algorithm

**NEAT** = NeuroEvolution of Augmenting Topologies

Unlike traditional neural networks, NEAT:
- **Evolves topology**: Networks start simple and grow complexity as needed
- **Speciation**: Protects innovation by grouping similar networks
- **Historical markings**: Tracks gene history for proper crossover

## 🏗️ Network Architecture

| Layer | Nodes | Description |
|-------|-------|-------------|
| **Inputs** | 12 | Obstacle 1 (5), Obstacle 2 (3), Dino State (4) |
| **Hidden** | Variable | Evolved by NEAT |
| **Outputs** | 2 | JUMP, DUCK (RUN is automatic/default) |

### Input Details
| # | Input | Description |
|---|-------|-------------|
| 0 | O1_Dist | Distance to obstacle 1 (normalized) |
| 1 | O1_Width | Width of obstacle 1 |
| 2 | O1_Height | Height of obstacle 1 |
| 3 | O1_Y | Y position of obstacle 1 |
| 4 | O1_Bird | Is obstacle 1 a bird? (0/1) |
| 5 | O2_Dist | Distance to obstacle 2 |
| 6 | O2_Height | Height of obstacle 2 |
| 7 | O2_Bird | Is obstacle 2 a bird? (0/1) |
| 8 | Dino_Y | Dino's Y position |
| 9 | Velocity | Dino's vertical velocity |
| 10 | Speed | Current game speed |
| 11 | Is_Duck | Is dino ducking? (0/1) |

### Output Logic
- **Threshold-based**: Action fires if output > 0.5
- **JUMP** (output 0): Dino jumps
- **DUCK** (output 1): Dino ducks
- **RUN**: Default action when neither output fires

## 🎮 Reward System

| Type | Points | Description |
|------|--------|-------------|
| Obstacle passed | +10 | Per obstacle cleared |
| Survival | +0.1/frame | Staying alive |
| Score milestone | +12/100pts | Bonus every 100 score |
| Score milestone | +15/500pts | Bonus every 500 score |
| **Bird penalty** | **-30/frame** | Not ducking when low/mid bird is near (50-150px) |

## 🛠️ Technical Stack

- **Python**: 3.11+
- **NEAT**: `neat-python` library
- **Game Engine**: Pygame
- **Visualization**: Matplotlib
- **Hardware**: CPU-based (no GPU required)

### Key Dependencies
```
neat-python
pygame
numpy
matplotlib
```

## 📋 Project Structure

```
Dino RL/
├── README.md                      # This file
├── state_based_training.ipynb     # Main training notebook
├── neat_config.txt                # NEAT configuration (auto-generated)
├── src/
│   ├── neat_trainer.py           # NEAT trainer, DinoGame, config generator
│   └── dino_game.py              # Original game implementation
├── models_state/
│   ├── neat_winner.pkl           # Best genome
│   ├── neat_stats.pkl            # Training statistics
│   └── neat_checkpoint.pkl       # Full checkpoint
├── plots_state/
│   ├── neat_fitness_per_gen.png  # Fitness per generation
│   ├── neat_best_over_time.png   # Cumulative best fitness
│   ├── neat_avg_trend.png        # Average fitness trend
│   └── neat_network_topology.png # Network visualization
└── prev_models/                   # Backup of previous models
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SaiGuruInukurthi/DINO-game-RL-trial-.git
cd DINO-game-RL-trial-
```

2. Install dependencies:
```bash
pip install neat-python pygame numpy matplotlib
```

3. Open the notebook:
```bash
jupyter notebook state_based_training.ipynb
```

## 📓 Using the Training Notebook

The notebook has several cells for different purposes:

### Cell 3: Configuration
```python
POPULATION_SIZE = 100      # Dinos per generation
GENERATIONS = 1            # Generations to train
START_SPEED = 8            # Starting game speed (default=14, max=25)
MUTATION_RATE = 0.75       # Weight/bias mutation rate (0.0-1.0)
ELITISM = 25               # Top genomes preserved each generation
```

### Cell 5: Fresh Training
Starts training from scratch (backs up old models first)

### Cell 7: Resume Training (Visual)
Continue training with Pygame visualization - slower but you can watch!

### Cell 9: Resume Training (Headless)
Continue training without visualization - **much faster!**
```python
HEADLESS_GENERATIONS = 1000
HEADLESS_POPULATION = 1000
HEADLESS_START_SPEED = 5
```

### Cell 11: Visualization
View training progress with 3 plots:
- **Fitness per Generation**: Best & average fitness each gen
- **Best Fitness Over Time**: Cumulative best (should only go up!)
- **Average Fitness Trend**: Rolling avg, raw avg, cumulative avg

### Cell 13: Test the Winner
Watch your best evolved Dino play!

### Cell 15: Network Topology
Visualize the evolved neural network structure

## ⚙️ Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 250 | Genomes per generation |
| `mutation_rate` | 0.7 | Weight/bias mutation probability (0.0-1.0) |
| `elitism` | 25 | Top genomes preserved unchanged |
| `start_speed` | 14 | Initial game speed (max 25) |

### Tips
- **Lower mutation rate** (0.3-0.5) = More stable, slower learning
- **Higher mutation rate** (0.7-0.9) = More exploration, can lose good solutions
- **Higher elitism** (30-50) = Safer, preserves more top performers
- **Lower start_speed** (5-10) = Easier training, then increase for challenge

## 📊 Training Progress

The model tracks:
- **Best Fitness**: Highest fitness in each generation
- **Average Fitness**: Mean fitness across population
- **Generation Count**: Preserved across training sessions

### Understanding the Plots
- **Raw Avg**: Actual average each generation (noisy)
- **Rolling Avg**: Smoothed over last 10 generations (recent trend)
- **Cumulative Avg**: All-time average (overall progress)

## 🎯 Game Details

### Obstacles
- **Cacti**: 50% spawn rate, various sizes
- **Birds**: 50% spawn rate
  - High birds: 20% (dino can run under)
  - **Middle birds**: 60% (must duck!)
  - Low birds: 20% (must jump)

### Speed Progression
- Game speed increases every 500 frames
- Max speed: 25
- Configurable start speed for training difficulty

## 🔄 Resuming Training

Training progress is saved automatically:
- `neat_winner.pkl`: Best genome ever
- `neat_stats.pkl`: All generation scores
- Generation count is preserved across sessions

Just run the **Resume Training** cell to continue!

## 📈 Success Metrics

- Best fitness > 3000+ (consistently avoids obstacles)
- Average fitness trending upward over generations
- Network evolves hidden nodes for complex decisions

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

## 📝 License

Open source - feel free to use and modify.

## 🔗 Resources

- [NEAT Paper](http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)
- [neat-python Documentation](https://neat-python.readthedocs.io/)
- [Chrome Dino Game](chrome://dino)

## 👨‍💻 Author

SaiGuruInukurthi

---

**Status**: 🚀 Active Development

Last Updated: November 29, 2025
