"""
Genetic Algorithm Trainer for Dino Game.
Trains 500 neural networks in parallel, selects the best, and evolves them.
Much faster than DQN for this type of game!
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from collections import defaultdict
import time

# Add paths
GAME_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Chrome-Dino-Runner')
sys.path.insert(0, GAME_DIR)

import pygame


class DinoNetwork(nn.Module):
    """
    Simple neural network for Dino game.
    Input: 8 state features
    Output: 3 action probabilities (run, jump, duck)
    """
    
    def __init__(self, input_size=8, hidden_size=16, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_action(self, state):
        """Get action from state (no gradient needed)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            output = self.forward(state_tensor)
            return output.argmax().item()
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        """Mutate network weights"""
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_strength
                param.add_(mask * noise)
    
    def copy(self):
        """Create a copy of this network"""
        new_net = DinoNetwork()
        new_net.load_state_dict(copy.deepcopy(self.state_dict()))
        return new_net


class DinoInstance:
    """
    A single Dino game instance (no rendering).
    Tracks one network playing the game.
    """
    
    # Game constants
    SCREEN_WIDTH = 1100
    DINO_X_POS = 80
    DINO_Y_POS = 310
    JUMP_VEL = 8.5
    
    def __init__(self, network):
        self.network = network
        self.reset()
        
    def reset(self):
        """Reset game state"""
        self.dino_y = self.DINO_Y_POS
        self.dino_jump = False
        self.dino_duck = False
        self.jump_vel = self.JUMP_VEL
        self.game_speed = 12
        self.score = 0
        self.alive = True
        self.obstacles = []
        self.frames_since_action = 0
        
        # Spawn first obstacle
        self._spawn_obstacle()
        
    def _spawn_obstacle(self):
        """Spawn a new obstacle"""
        if len(self.obstacles) < 2:
            obs_type = random.choice(['small_cactus', 'large_cactus', 'bird'])
            
            if obs_type == 'small_cactus':
                self.obstacles.append({
                    'x': self.SCREEN_WIDTH + random.randint(0, 200),
                    'y': 325,
                    'width': 34,
                    'height': 70,
                    'is_bird': False
                })
            elif obs_type == 'large_cactus':
                self.obstacles.append({
                    'x': self.SCREEN_WIDTH + random.randint(0, 200),
                    'y': 300,
                    'width': 50,
                    'height': 95,
                    'is_bird': False
                })
            else:  # bird
                bird_y = random.choice([280, 310, 340])  # Different heights
                self.obstacles.append({
                    'x': self.SCREEN_WIDTH + random.randint(0, 200),
                    'y': bird_y,
                    'width': 60,
                    'height': 50,
                    'is_bird': True
                })
    
    def get_state(self):
        """Extract state features"""
        state = np.zeros(8, dtype=np.float32)
        
        if self.obstacles:
            sorted_obs = sorted(self.obstacles, key=lambda o: o['x'])
            nearest = sorted_obs[0]
            
            state[0] = np.clip((nearest['x'] - self.DINO_X_POS) / self.SCREEN_WIDTH, 0, 1)
            state[1] = np.clip((self.DINO_Y_POS - nearest['y']) / 100, 0, 1)
            state[2] = np.clip(nearest['width'] / 100, 0, 1)
            state[7] = 1.0 if nearest['is_bird'] else 0.0
            
            if len(sorted_obs) > 1:
                state[6] = np.clip((sorted_obs[1]['x'] - self.DINO_X_POS) / self.SCREEN_WIDTH, 0, 1)
            else:
                state[6] = 1.0
        else:
            state[0] = 1.0
            state[6] = 1.0
            
        state[3] = np.clip((self.DINO_Y_POS - self.dino_y) / 150, 0, 1)
        state[4] = 1.0 if self.dino_jump else 0.0
        state[5] = np.clip(self.game_speed / 40, 0, 1)
        
        return state
    
    def step(self):
        """Run one game step"""
        if not self.alive:
            return
        
        # Get action from network
        state = self.get_state()
        action = self.network.get_action(state)
        
        # Apply action (with cooldown)
        self.frames_since_action += 1
        if self.frames_since_action >= 3:  # Action cooldown
            if action == 1 and not self.dino_jump:  # Jump
                self.dino_jump = True
                self.dino_duck = False
                self.frames_since_action = 0
            elif action == 2 and not self.dino_jump:  # Duck
                self.dino_duck = True
                self.frames_since_action = 0
            elif action == 0:  # Run
                if not self.dino_jump:
                    self.dino_duck = False
        
        # Update dino physics
        if self.dino_jump:
            self.dino_y -= self.jump_vel * 4
            self.jump_vel -= 0.8
            if self.jump_vel < -self.JUMP_VEL:
                self.dino_jump = False
                self.jump_vel = self.JUMP_VEL
                self.dino_y = self.DINO_Y_POS
        
        # Update obstacles
        for obs in self.obstacles[:]:
            obs['x'] -= self.game_speed
            if obs['x'] < -100:
                self.obstacles.remove(obs)
        
        # Spawn new obstacles
        if len(self.obstacles) == 0 or (
            len(self.obstacles) < 2 and 
            self.obstacles[-1]['x'] < self.SCREEN_WIDTH - 300
        ):
            self._spawn_obstacle()
        
        # Check collision
        dino_rect = pygame.Rect(
            self.DINO_X_POS + 5, 
            self.dino_y + 5 if not self.dino_duck else self.dino_y + 30,
            40, 
            80 if not self.dino_duck else 50
        )
        
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['x'] + 5, obs['y'] + 5, obs['width'] - 10, obs['height'] - 10)
            if dino_rect.colliderect(obs_rect):
                self.alive = False
                return
        
        # Update score and speed
        self.score += 1
        if self.score % 100 == 0 and self.game_speed < 20:
            self.game_speed += 0.5


class GeneticTrainer:
    """
    Genetic Algorithm trainer for Dino game.
    Evolves a population of neural networks.
    """
    
    def __init__(
        self,
        population_size=500,
        elite_count=50,
        mutation_rate=0.1,
        mutation_strength=0.3,
        max_steps=5000,
        render=True  # NEW: Show visual of best dino
    ):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.max_steps = max_steps
        self.render = render
        
        # Initialize population
        self.population = [DinoNetwork() for _ in range(population_size)]
        self.generation = 0
        self.best_score_ever = 0
        self.best_network = None
        
        # Stats
        self.generation_stats = []
        
        # Visual rendering setup
        if self.render:
            os.chdir(GAME_DIR)
            pygame.init()
            self.screen = pygame.display.set_mode((1100, 600))
            pygame.display.set_caption("🧬 Dino Genetic Evolution - LIVE")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font("freesansbold.ttf", 20)
            self._load_assets()
        
        print(f"🧬 Genetic Trainer initialized")
        print(f"   Population: {population_size}")
        print(f"   Elites: {elite_count}")
        print(f"   Mutation rate: {mutation_rate}")
        print(f"   Max steps per game: {max_steps}")
        print(f"   Visual: {'ON 🎮' if render else 'OFF'}")
    
    def _load_assets(self):
        """Load game sprites for rendering"""
        self.RUNNING = [
            pygame.image.load(os.path.join("assets/Dino", "DinoRun1.png")),
            pygame.image.load(os.path.join("assets/Dino", "DinoRun2.png")),
        ]
        self.JUMPING = pygame.image.load(os.path.join("assets/Dino", "DinoJump.png"))
        self.DUCKING = [
            pygame.image.load(os.path.join("assets/Dino", "DinoDuck1.png")),
            pygame.image.load(os.path.join("assets/Dino", "DinoDuck2.png")),
        ]
        self.SMALL_CACTUS = [
            pygame.image.load(os.path.join("assets/Cactus", "SmallCactus1.png")),
            pygame.image.load(os.path.join("assets/Cactus", "SmallCactus2.png")),
            pygame.image.load(os.path.join("assets/Cactus", "SmallCactus3.png")),
        ]
        self.LARGE_CACTUS = [
            pygame.image.load(os.path.join("assets/Cactus", "LargeCactus1.png")),
            pygame.image.load(os.path.join("assets/Cactus", "LargeCactus2.png")),
            pygame.image.load(os.path.join("assets/Cactus", "LargeCactus3.png")),
        ]
        self.BIRD = [
            pygame.image.load(os.path.join("assets/Bird", "Bird1.png")),
            pygame.image.load(os.path.join("assets/Bird", "Bird2.png")),
        ]
        self.BG = pygame.image.load(os.path.join("assets/Other", "Track.png"))
    
    def _render_frame(self, lead_instance, alive_count, step):
        """Render the leading dino"""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Background
        self.screen.fill((255, 255, 255))
        
        # Track - use score as proxy for distance traveled
        x_pos_bg = -(lead_instance.score * 3) % self.BG.get_width()
        self.screen.blit(self.BG, (x_pos_bg, 380))
        self.screen.blit(self.BG, (x_pos_bg + self.BG.get_width(), 380))
        
        step_index = (step // 5) % 2
        
        # Dino
        dino_y = int(lead_instance.dino_y)
        if lead_instance.dino_jump:
            self.screen.blit(self.JUMPING, (80, dino_y))
        elif lead_instance.dino_duck:
            self.screen.blit(self.DUCKING[step_index], (80, 340))
        else:
            self.screen.blit(self.RUNNING[step_index], (80, dino_y))
        
        # Obstacles - only render those on screen
        for obs in lead_instance.obstacles:
            x = int(obs['x'])
            y = int(obs['y'])
            
            # Only render if on screen
            if -100 < x < 1200:
                if obs['is_bird']:
                    self.screen.blit(self.BIRD[step_index], (x, y))
                elif obs['height'] > 80:
                    self.screen.blit(self.LARGE_CACTUS[0], (x, y))
                else:
                    self.screen.blit(self.SMALL_CACTUS[0], (x, y))
        
        # Info overlay
        info_text = self.font.render(
            f"Gen: {self.generation} | Score: {lead_instance.score} | Alive: {alive_count}/{self.population_size} | Best Ever: {self.best_score_ever}",
            True, (0, 0, 0)
        )
        self.screen.blit(info_text, (10, 10))
        
        status = "LEADING DINO" if lead_instance.alive else "DIED - Showing replay"
        status_color = (0, 128, 0) if lead_instance.alive else (255, 0, 0)
        status_text = self.font.render(status, True, status_color)
        self.screen.blit(status_text, (10, 35))
        
        pygame.display.update()
        self.clock.tick(60)
        return True
    
    def run_generation(self, render_best=None):
        """Run one generation of evolution"""
        if render_best is None:
            render_best = self.render
            
        self.generation += 1
        start_time = time.time()
        
        # Create game instances for each network
        instances = [DinoInstance(net) for net in self.population]
        
        # Run all games simultaneously
        step = 0
        alive_count = len(instances)
        lead_idx = 0  # Track the leading dino
        
        while alive_count > 0 and step < self.max_steps:
            alive_count = 0
            best_score_this_step = -1
            
            for i, inst in enumerate(instances):
                if inst.alive:
                    inst.step()
                    alive_count += 1
                    # Track who's in the lead
                    if inst.score > best_score_this_step:
                        best_score_this_step = inst.score
                        lead_idx = i
            
            step += 1
            
            # Render the leading dino
            if render_best and step % 1 == 0:  # Every frame
                if not self._render_frame(instances[lead_idx], alive_count, step):
                    # User closed window
                    self.render = False
                    render_best = False
        
        print(" " * 80, end='\r')
        
        # Get scores
        scores = [(inst.score, i) for i, inst in enumerate(instances)]
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Stats
        all_scores = [s[0] for s in scores]
        best_score = scores[0][0]
        avg_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        
        # Track best ever
        if best_score > self.best_score_ever:
            self.best_score_ever = best_score
            self.best_network = self.population[scores[0][1]].copy()
        
        elapsed = time.time() - start_time
        
        # Log stats
        stats = {
            'generation': self.generation,
            'best_score': best_score,
            'avg_score': avg_score,
            'median_score': median_score,
            'best_ever': self.best_score_ever,
            'time': elapsed
        }
        self.generation_stats.append(stats)
        
        print(f"🧬 Gen {self.generation:3d} | Best: {best_score:5.0f} | Avg: {avg_score:5.0f} | "
              f"Median: {median_score:5.0f} | Best Ever: {self.best_score_ever:5.0f} | Time: {elapsed:.1f}s")
        
        # Selection and reproduction
        self._evolve(scores)
        
        return stats
    
    def _evolve(self, scores):
        """Select best networks and create next generation"""
        # Get elite indices
        elite_indices = [s[1] for s in scores[:self.elite_count]]
        elites = [self.population[i].copy() for i in elite_indices]
        
        # Create new population
        new_population = []
        
        # Keep elites unchanged
        new_population.extend(elites)
        
        # Fill rest with mutated copies of elites
        while len(new_population) < self.population_size:
            # Select parent (weighted by rank)
            parent_idx = random.choices(
                range(len(elites)),
                weights=[len(elites) - i for i in range(len(elites))]
            )[0]
            
            child = elites[parent_idx].copy()
            child.mutate(self.mutation_rate, self.mutation_strength)
            new_population.append(child)
        
        self.population = new_population
    
    def save_best(self, filepath):
        """Save the best network"""
        if self.best_network:
            torch.save({
                'network': self.best_network.state_dict(),
                'generation': self.generation,
                'best_score': self.best_score_ever,
                'stats': self.generation_stats
            }, filepath)
            print(f"💾 Best network saved: {filepath}")
    
    def load(self, filepath):
        """Load a saved network"""
        checkpoint = torch.load(filepath)
        self.best_network = DinoNetwork()
        self.best_network.load_state_dict(checkpoint['network'])
        self.generation = checkpoint.get('generation', 0)
        self.best_score_ever = checkpoint.get('best_score', 0)
        self.generation_stats = checkpoint.get('stats', [])
        
        # Initialize population from best network
        self.population = []
        for _ in range(self.population_size):
            net = self.best_network.copy()
            if _ > 0:  # Don't mutate the first one
                net.mutate(self.mutation_rate, self.mutation_strength)
            self.population.append(net)
        
        print(f"📂 Loaded from gen {self.generation}, best score: {self.best_score_ever}")
    
    def close(self):
        """Clean up pygame"""
        if self.render:
            pygame.quit()


class VisualTester:
    """
    Visual tester to watch the best network play.
    """
    
    def __init__(self, network):
        self.network = network
        
        # Initialize pygame
        os.chdir(GAME_DIR)
        pygame.init()
        self.screen = pygame.display.set_mode((1100, 600))
        pygame.display.set_caption("Dino GA - Best Network")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("freesansbold.ttf", 20)
        
        # Load assets
        self._load_assets()
        
    def _load_assets(self):
        """Load game sprites"""
        self.RUNNING = [
            pygame.image.load(os.path.join("assets/Dino", "DinoRun1.png")),
            pygame.image.load(os.path.join("assets/Dino", "DinoRun2.png")),
        ]
        self.JUMPING = pygame.image.load(os.path.join("assets/Dino", "DinoJump.png"))
        self.DUCKING = [
            pygame.image.load(os.path.join("assets/Dino", "DinoDuck1.png")),
            pygame.image.load(os.path.join("assets/Dino", "DinoDuck2.png")),
        ]
        self.SMALL_CACTUS = [
            pygame.image.load(os.path.join("assets/Cactus", "SmallCactus1.png")),
            pygame.image.load(os.path.join("assets/Cactus", "SmallCactus2.png")),
            pygame.image.load(os.path.join("assets/Cactus", "SmallCactus3.png")),
        ]
        self.LARGE_CACTUS = [
            pygame.image.load(os.path.join("assets/Cactus", "LargeCactus1.png")),
            pygame.image.load(os.path.join("assets/Cactus", "LargeCactus2.png")),
            pygame.image.load(os.path.join("assets/Cactus", "LargeCactus3.png")),
        ]
        self.BIRD = [
            pygame.image.load(os.path.join("assets/Bird", "Bird1.png")),
            pygame.image.load(os.path.join("assets/Bird", "Bird2.png")),
        ]
        self.BG = pygame.image.load(os.path.join("assets/Other", "Track.png"))
        
    def run(self, num_games=5):
        """Run visual test"""
        scores = []
        
        for game in range(num_games):
            instance = DinoInstance(self.network)
            step_index = 0
            x_pos_bg = 0
            
            running = True
            while running and instance.alive:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return scores
                
                instance.step()
                step_index = (step_index + 1) % 10
                
                # Render
                self.screen.fill((255, 255, 255))
                
                # Background
                x_pos_bg -= instance.game_speed
                if x_pos_bg <= -self.BG.get_width():
                    x_pos_bg = 0
                self.screen.blit(self.BG, (x_pos_bg, 380))
                self.screen.blit(self.BG, (x_pos_bg + self.BG.get_width(), 380))
                
                # Dino
                if instance.dino_jump:
                    self.screen.blit(self.JUMPING, (80, instance.dino_y))
                elif instance.dino_duck:
                    self.screen.blit(self.DUCKING[step_index // 5], (80, 340))
                else:
                    self.screen.blit(self.RUNNING[step_index // 5], (80, instance.dino_y))
                
                # Obstacles
                for obs in instance.obstacles:
                    if obs['is_bird']:
                        self.screen.blit(self.BIRD[step_index // 5], (obs['x'], obs['y']))
                    elif obs['height'] > 80:
                        self.screen.blit(self.LARGE_CACTUS[0], (obs['x'], obs['y']))
                    else:
                        self.screen.blit(self.SMALL_CACTUS[0], (obs['x'], obs['y']))
                
                # Score
                text = self.font.render(f"Score: {instance.score}  Game: {game+1}/{num_games}", True, (0, 0, 0))
                self.screen.blit(text, (10, 10))
                
                pygame.display.update()
                self.clock.tick(60)
            
            scores.append(instance.score)
            print(f"Game {game+1}: Score = {instance.score}")
            pygame.time.wait(500)
        
        pygame.quit()
        return scores


if __name__ == "__main__":
    # Quick test
    trainer = GeneticTrainer(population_size=100, elite_count=10)
    
    for gen in range(5):
        trainer.run_generation()
    
    trainer.save_best('models_state/genetic_best.pth')
