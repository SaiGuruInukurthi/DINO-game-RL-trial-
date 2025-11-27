"""
NEAT (NeuroEvolution of Augmenting Topologies) Trainer for Chrome Dino Game

NEAT evolves both the network TOPOLOGY and WEIGHTS:
- Starts with minimal networks (just input→output)
- Adds neurons and connections through mutation
- Uses speciation to protect innovation
- Historical markings prevent competing conventions problem

This is more powerful than fixed-topology evolution!
"""

import neat
import pygame
import numpy as np
import os
import pickle
import random
import copy
from datetime import datetime

# Game constants (same as Chrome Dino Runner)
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 600
GROUND_Y = 310  # Match original game

# Asset path for sprites
ASSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Chrome-Dino-Runner", "assets")

# Global sprites (loaded once)
SPRITES = None

def load_sprites():
    """Load all game sprites from Chrome-Dino-Runner"""
    global SPRITES
    if SPRITES is not None:
        return SPRITES
    
    SPRITES = {}
    try:
        # Dino sprites
        SPRITES['RUNNING'] = [
            pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoRun1.png")),
            pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoRun2.png")),
        ]
        SPRITES['JUMPING'] = pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoJump.png"))
        SPRITES['DUCKING'] = [
            pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoDuck1.png")),
            pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoDuck2.png")),
        ]
        
        # Cactus sprites
        SPRITES['SMALL_CACTUS'] = [
            pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "SmallCactus1.png")),
            pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "SmallCactus2.png")),
            pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "SmallCactus3.png")),
        ]
        SPRITES['LARGE_CACTUS'] = [
            pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "LargeCactus1.png")),
            pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "LargeCactus2.png")),
            pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "LargeCactus3.png")),
        ]
        
        # Bird sprites
        SPRITES['BIRD'] = [
            pygame.image.load(os.path.join(ASSET_PATH, "Bird", "Bird1.png")),
            pygame.image.load(os.path.join(ASSET_PATH, "Bird", "Bird2.png")),
        ]
        
        # Background
        SPRITES['BG'] = pygame.image.load(os.path.join(ASSET_PATH, "Other", "Track.png"))
        SPRITES['CLOUD'] = pygame.image.load(os.path.join(ASSET_PATH, "Other", "Cloud.png"))
        
        print("✓ Loaded original Chrome Dino sprites!")
    except Exception as e:
        print(f"⚠ Could not load sprites: {e}")
        SPRITES = None
    
    return SPRITES


class DinoGame:
    """
    Headless Dino game logic for NEAT evaluation.
    Each genome gets its own game instance.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Dino state
        self.dino_y = GROUND_Y
        self.dino_vel_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.dino_height = 50  # Normal height
        self.duck_height = 30  # Ducking height
        
        # Game state
        self.score = 0  # Time-based score (like real Chrome Dino)
        self.obstacles_passed = 0  # Count of obstacles passed (for reward)
        self.game_speed = 14
        self.game_over = False
        self.steps = 0
        
        # Obstacles
        self.obstacles = []
        self.spawn_obstacle()
        
        return self.get_state()
    
    def spawn_obstacle(self):
        """Spawn a cactus or bird"""
        obstacle_type = random.choice(['cactus', 'cactus', 'cactus', 'bird'])  # 75% cactus
        
        if obstacle_type == 'cactus':
            # Cactus variations
            width = random.choice([20, 40, 60])
            height = random.choice([40, 50, 70])
            self.obstacles.append({
                'type': 'cactus',
                'x': SCREEN_WIDTH + 50,
                'y': 380 - height,  # Bottom of cactus sits on ground (y=380)
                'width': width,
                'height': height
            })
        else:
            # Bird at different heights
            bird_y = random.choice([GROUND_Y - 60, GROUND_Y - 100, GROUND_Y - 140])
            self.obstacles.append({
                'type': 'bird',
                'x': SCREEN_WIDTH + 50,
                'y': bird_y,
                'width': 60,
                'height': 40
            })
    
    def get_state(self):
        """
        Get normalized state for neural network.
        12 inputs: info about 2 nearest obstacles + dino state.
        """
        dino_x = 80  # Dino x position
        
        # Find the 2 nearest obstacles (sorted by distance)
        obstacles_ahead = []
        for obs in self.obstacles:
            dist = obs['x'] - dino_x
            if dist > -50:  # Include obstacles slightly behind (still dangerous)
                obstacles_ahead.append((dist, obs))
        
        # Sort by distance and take nearest 2
        obstacles_ahead.sort(key=lambda x: x[0])
        
        # Get nearest obstacle (or default values if none)
        if len(obstacles_ahead) >= 1:
            dist1, obs1 = obstacles_ahead[0]
            o1_dist = dist1 / SCREEN_WIDTH
            o1_width = obs1['width'] / 100.0
            o1_height = obs1['height'] / 100.0
            o1_y = (obs1['y'] - (GROUND_Y - 100)) / 100.0
            o1_bird = 1.0 if obs1['type'] == 'bird' else 0.0
        else:
            # No obstacles - safe defaults
            o1_dist, o1_width, o1_height, o1_y, o1_bird = 1.0, 0.0, 0.0, 0.0, 0.0
        
        # Get second nearest obstacle (or default values if none)
        if len(obstacles_ahead) >= 2:
            dist2, obs2 = obstacles_ahead[1]
            o2_dist = dist2 / SCREEN_WIDTH
            o2_height = obs2['height'] / 100.0
            o2_bird = 1.0 if obs2['type'] == 'bird' else 0.0
        else:
            # No second obstacle - far away defaults
            o2_dist, o2_height, o2_bird = 1.0, 0.0, 0.0
        
        state = np.array([
            # Obstacle 1 (nearest)
            o1_dist,                                        # 0: Distance to obstacle 1
            o1_width,                                       # 1: Obstacle 1 width
            o1_height,                                      # 2: Obstacle 1 height
            o1_y,                                           # 3: Obstacle 1 y position
            o1_bird,                                        # 4: Is obstacle 1 a bird?
            # Obstacle 2 (second nearest)
            o2_dist,                                        # 5: Distance to obstacle 2
            o2_height,                                      # 6: Obstacle 2 height
            o2_bird,                                        # 7: Is obstacle 2 a bird?
            # Dino state
            (self.dino_y - GROUND_Y) / 100.0,               # 8: Dino y position
            self.dino_vel_y / 20.0,                         # 9: Dino velocity
            self.game_speed / 20.0,                         # 10: Game speed
            1.0 if self.is_ducking else 0.0                 # 11: Is dino ducking?
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        Execute action and return (state, reward, done).
        Action: 0=RUN, 1=JUMP, 2=DUCK
        """
        if self.game_over:
            return self.get_state(), 0, True
        
        self.steps += 1
        reward = 0.25  # Small reward for surviving
        
        # Handle action
        if action == 1 and not self.is_jumping and self.dino_y >= GROUND_Y:
            # JUMP
            self.is_jumping = True
            self.is_ducking = False
            self.dino_vel_y = -16
        elif action == 2:
            # DUCK (only on ground)
            if not self.is_jumping:
                self.is_ducking = True
        else:
            # RUN
            if not self.is_jumping:
                self.is_ducking = False
        
        # Physics
        if self.is_jumping or self.dino_y < GROUND_Y:
            self.dino_vel_y += 0.8  # Gravity
            self.dino_y += self.dino_vel_y
            
            if self.dino_y >= GROUND_Y:
                self.dino_y = GROUND_Y
                self.is_jumping = False
                self.dino_vel_y = 0
        
        # Move obstacles
        for obs in self.obstacles:
            obs['x'] -= self.game_speed
        
        # Update time-based score (10 points per second at 60fps)
        if self.steps % 6 == 0:
            self.score += 1
        
        # Remove passed obstacles and count them for reward
        passed = [obs for obs in self.obstacles if obs['x'] < -100]
        if passed:
            reward += 10  # Bonus for passing obstacle
            self.obstacles_passed += len(passed)
        
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -100]
        
        # Spawn new obstacles
        if len(self.obstacles) == 0 or self.obstacles[-1]['x'] < SCREEN_WIDTH - 300:
            if random.random() < 0.02 or len(self.obstacles) == 0:
                self.spawn_obstacle()
        
        # Collision detection
        dino_x = 80
        current_height = self.duck_height if self.is_ducking else self.dino_height
        dino_rect = (dino_x, self.dino_y - current_height + 50, 40, current_height)
        
        for obs in self.obstacles:
            obs_rect = (obs['x'], obs['y'], obs['width'], obs['height'])
            if self._check_collision(dino_rect, obs_rect):
                self.game_over = True
                reward = -100  # Big penalty for dying
                break
        
        # Increase speed gradually
        if self.steps % 500 == 0:
            self.game_speed = min(25, self.game_speed + 0.5)
        
        return self.get_state(), reward, self.game_over
    
    def _check_collision(self, rect1, rect2):
        """Simple AABB collision with some padding"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Add padding for forgiveness
        padding = 5
        return (x1 + padding < x2 + w2 - padding and
                x1 + w1 - padding > x2 + padding and
                y1 + padding < y2 + h2 - padding and
                y1 + h1 - padding > y2 + padding)


class NEATTrainer:
    """
    NEAT-based trainer for Chrome Dino game.
    
    Key NEAT features:
    - Starts with minimal networks
    - Evolves topology through structural mutations
    - Speciation protects new innovations
    - Historical markings track gene origins
    """
    
    def __init__(self, config_path, render=True):
        self.config_path = config_path
        self.render = render
        self.generation = 0
        self.best_genome = None
        self.best_fitness = 0
        
        # Stats tracking
        self.gen_best_scores = []
        self.gen_avg_scores = []
        
        # Load NEAT config
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Create population
        self.population = neat.Population(self.config)
        
        # Add statistics reporter (no verbose stdout)
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        
        # Pygame for visualization
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("🧬 NEAT Dino Evolution")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
            
            # Load sprites
            self.sprites = load_sprites()
            self.step_index = 0  # For animation
            self.x_pos_bg = 0    # Background scroll position
    
    def eval_genomes_parallel(self, genomes, config):
        """
        Evaluate ALL genomes simultaneously with visual rendering.
        All 250 dinos run at the same time on screen!
        """
        import time
        gen_start_time = time.time()
        self.generation += 1
        
        # Create networks and games for ALL genomes
        dinos = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            game = DinoGame()
            game.reset()
            # Assign a unique color to each dino based on genome_id
            hue = (genome_id * 37) % 360  # Spread colors
            color = self._hue_to_rgb(hue)
            dinos.append({
                'genome': genome,
                'genome_id': genome_id,
                'net': net,
                'game': game,
                'alive': True,
                'color': color,
                'fitness': 0,
                'bird_penalty': 0  # Penalty for not ducking when bird approaches
            })
        
        # Shared obstacle system for synchronized gameplay
        shared_obstacles = []
        shared_game_speed = 14
        shared_steps = 0
        shared_score = 0  # Time-based score (like real Chrome Dino)
        obstacles_passed = 0  # Count of obstacles passed
        
        # Spawn initial obstacle
        self._spawn_shared_obstacle(shared_obstacles)
        
        max_steps = 50000  # Increased from 5000 to allow longer runs
        
        while any(d['alive'] for d in dinos) and shared_steps < max_steps:
            shared_steps += 1
            
            # Handle pygame events
            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            
            # Move obstacles
            for obs in shared_obstacles:
                obs['x'] -= shared_game_speed
            
            # Update time-based score (10 points per second at 60fps)
            shared_score += 1 if shared_steps % 6 == 0 else 0
            
            # Check for passed obstacles (for reward calculation)
            passed = [obs for obs in shared_obstacles if obs['x'] < -100]
            if passed:
                obstacles_passed += len(passed)
            
            shared_obstacles = [obs for obs in shared_obstacles if obs['x'] > -100]
            
            # Spawn new obstacles
            if len(shared_obstacles) == 0 or shared_obstacles[-1]['x'] < SCREEN_WIDTH - 300:
                if random.random() < 0.02 or len(shared_obstacles) == 0:
                    self._spawn_shared_obstacle(shared_obstacles)
            
            # Increase speed gradually
            if shared_steps % 500 == 0:
                shared_game_speed = min(25, shared_game_speed + 0.5)
            
            # Update each dino
            for dino in dinos:
                if not dino['alive']:
                    continue
                
                game = dino['game']
                net = dino['net']
                
                # Sync obstacles to dino's game
                game.obstacles = [obs.copy() for obs in shared_obstacles]
                game.game_speed = shared_game_speed
                game.steps = shared_steps
                
                # Get state and action
                state = game.get_state()
                output = net.activate(state)
                # 2 outputs: [JUMP, DUCK] - use thresholds
                # If JUMP output > 0.5: jump, elif DUCK output > 0.5: duck, else: run
                if output[0] > 0.5 and output[0] > output[1]:
                    action = 1  # JUMP
                elif output[1] > 0.5:
                    action = 2  # DUCK
                else:
                    action = 0  # RUN (default)
                
                # Execute action (physics only, obstacles already synced)
                self._step_dino_physics(game, action)
                
                # Check collision
                if self._check_dino_collision(game, shared_obstacles):
                    dino['alive'] = False
                    # Calculate tiered reward bonuses
                    # +12 for every 100 score, +15 for every 500 score
                    bonus_100 = (shared_score // 100) * 12
                    bonus_500 = (shared_score // 500) * 15
                    milestone_bonus = bonus_100 + bonus_500
                    reward = (obstacles_passed * 10) + (shared_steps * 0.1) + milestone_bonus
                    dino['fitness'] = reward
                    dino['obstacles_passed'] = obstacles_passed
            
            # Render all dinos
            if self.render:
                self._render_all_dinos(dinos, shared_obstacles, shared_score, shared_game_speed, shared_steps, obstacles_passed)
        
        # Set fitness for all genomes
        best_fitness_in_gen = 0
        best_genome_in_gen = None
        scores = []
        
        for dino in dinos:
            # If still alive, give them full score
            if dino['alive']:
                # Calculate tiered reward bonuses
                # +12 for every 100 score, +15 for every 500 score
                bonus_100 = (shared_score // 100) * 12
                bonus_500 = (shared_score // 500) * 15
                milestone_bonus = bonus_100 + bonus_500
                # Apply bird penalty for not ducking
                reward = (obstacles_passed * 10) + (shared_steps * 0.1) + milestone_bonus - dino['bird_penalty']
                dino['fitness'] = max(0, reward)  # Don't go negative
                dino['obstacles_passed'] = obstacles_passed
            
            dino['genome'].fitness = dino['fitness']
            scores.append(dino['fitness'])
            
            if dino['fitness'] > best_fitness_in_gen:
                best_fitness_in_gen = dino['fitness']
                best_genome_in_gen = dino['genome']
        
        # Track stats
        self.gen_best_scores.append(best_fitness_in_gen)
        self.gen_avg_scores.append(np.mean(scores))
        
        # Update global best - DEEP COPY to preserve it across generations!
        if best_fitness_in_gen > self.best_fitness:
            self.best_fitness = best_fitness_in_gen
            self.best_genome = copy.deepcopy(best_genome_in_gen)
            new_best = " ⭐NEW BEST!"
        else:
            new_best = ""
        
        # Calculate generation time
        gen_time = time.time() - gen_start_time
        
        # Get species count from population
        num_species = len(self.population.species.species) if hasattr(self, 'population') else '?'
        
        print(f"  Gen {self.generation:3d} | Best: {best_fitness_in_gen:5.0f} | Avg: {np.mean(scores):5.0f} | Best Ever: {self.best_fitness:5.0f} | Species: {num_species} | {gen_time:.1f}s{new_best}")
    
    def _hue_to_rgb(self, hue):
        """Convert hue (0-360) to RGB color"""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.9)
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def _spawn_shared_obstacle(self, obstacles):
        """Spawn obstacle to shared obstacle list"""
        obstacle_type = random.choice(['cactus', 'cactus', 'cactus', 'bird'])
        
        if obstacle_type == 'cactus':
            width = random.choice([20, 40, 60])
            height = random.choice([40, 50, 70])
            obstacles.append({
                'type': 'cactus',
                'x': SCREEN_WIDTH + 50,
                'y': 380 - height,  # Bottom of cactus sits on ground (y=380)
                'width': width,
                'height': height
            })
        else:
            bird_y = random.choice([GROUND_Y - 60, GROUND_Y - 100, GROUND_Y - 140])
            obstacles.append({
                'type': 'bird',
                'x': SCREEN_WIDTH + 50,
                'y': bird_y,
                'width': 60,
                'height': 40
            })
    
    def _step_dino_physics(self, game, action):
        """Apply physics and action to a dino (no obstacle movement)"""
        # Handle action
        if action == 1 and not game.is_jumping and game.dino_y >= GROUND_Y:
            game.is_jumping = True
            game.is_ducking = False
            game.dino_vel_y = -16
        elif action == 2:
            if not game.is_jumping:
                game.is_ducking = True
        else:
            if not game.is_jumping:
                game.is_ducking = False
        
        # Physics
        if game.is_jumping or game.dino_y < GROUND_Y:
            game.dino_vel_y += 0.8
            game.dino_y += game.dino_vel_y
            
            if game.dino_y >= GROUND_Y:
                game.dino_y = GROUND_Y
                game.is_jumping = False
                game.dino_vel_y = 0
    
    def _check_dino_collision(self, game, obstacles):
        """Check if dino collides with any obstacle"""
        dino_x = 80
        current_height = game.duck_height if game.is_ducking else game.dino_height
        dino_rect = (dino_x, game.dino_y - current_height + 50, 40, current_height)
        
        for obs in obstacles:
            obs_rect = (obs['x'], obs['y'], obs['width'], obs['height'])
            if self._aabb_collision(dino_rect, obs_rect):
                return True
        return False
    
    def _aabb_collision(self, rect1, rect2):
        """AABB collision with padding"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        padding = 5
        return (x1 + padding < x2 + w2 - padding and
                x1 + w1 - padding > x2 + padding and
                y1 + padding < y2 + h2 - padding and
                y1 + h1 - padding > y2 + padding)
    
    def _render_all_dinos(self, dinos, obstacles, score, game_speed, steps, obstacles_passed=0):
        """Render all dinos and shared obstacles"""
        self.screen.fill((255, 255, 255))
        
        # Draw background track
        if self.sprites and 'BG' in self.sprites:
            bg = self.sprites['BG']
            image_width = bg.get_width()
            self.screen.blit(bg, (self.x_pos_bg, 380))
            self.screen.blit(bg, (image_width + self.x_pos_bg, 380))
            if self.x_pos_bg <= -image_width:
                self.x_pos_bg = 0
            self.x_pos_bg -= game_speed
        else:
            pygame.draw.line(self.screen, (0, 0, 0), (0, 380), (SCREEN_WIDTH, 380), 2)
        
        # Update animation step
        self.step_index += 1
        if self.step_index >= 10:
            self.step_index = 0
        
        # Draw obstacles first
        for obs in obstacles:
            if obs['x'] > -100 and obs['x'] < SCREEN_WIDTH + 100:
                if self.sprites:
                    if obs['type'] == 'cactus':
                        if obs['height'] > 55:
                            cactus_img = self.sprites['LARGE_CACTUS'][0]
                        else:
                            cactus_img = self.sprites['SMALL_CACTUS'][0]
                        self.screen.blit(cactus_img, (obs['x'], obs['y']))
                    else:
                        bird_img = self.sprites['BIRD'][self.step_index // 5]
                        self.screen.blit(bird_img, (obs['x'], obs['y']))
                else:
                    color = (100, 100, 100) if obs['type'] == 'cactus' else (200, 50, 50)
                    pygame.draw.rect(self.screen, color, (obs['x'], obs['y'], obs['width'], obs['height']))
        
        # Count alive dinos
        alive_count = sum(1 for d in dinos if d['alive'])
        
        # Draw ALL alive dinos (with slight transparency effect using colored dinos)
        dino_x = 80
        for dino in dinos:
            if not dino['alive']:
                continue
            
            game = dino['game']
            color = dino['color']
            
            if self.sprites:
                # Get appropriate sprite
                if game.is_jumping:
                    dino_img = self.sprites['JUMPING'].copy()
                    y_pos = game.dino_y - 50
                elif game.is_ducking:
                    dino_img = self.sprites['DUCKING'][self.step_index // 5].copy()
                    y_pos = GROUND_Y + 30
                else:
                    dino_img = self.sprites['RUNNING'][self.step_index // 5].copy()
                    y_pos = GROUND_Y
                
                # Tint the sprite with dino's unique color
                tinted = self._tint_sprite(dino_img, color)
                self.screen.blit(tinted, (dino_x, y_pos))
            else:
                # Fallback: colored rectangle
                current_height = game.duck_height if game.is_ducking else game.dino_height
                pygame.draw.rect(self.screen, color,
                               (dino_x, game.dino_y - current_height + 50, 40, current_height))
        
        # Calculate current reward for display
        # Tiered bonuses: +12 per 100 score, +15 per 500 score
        bonus_100 = (score // 100) * 12
        bonus_500 = (score // 500) * 15
        milestone_bonus = bonus_100 + bonus_500
        current_reward = (obstacles_passed * 10) + (steps * 0.1) + milestone_bonus
        
        # Draw info overlay
        info_texts = [
            f"🧬 Gen: {self.generation} | Species: {len(self.population.species.species)}",
            f"Alive: {alive_count}/{len(dinos)}",
            f"Score: {score:,}",  # Time-based score (like real game)
            f"Reward: {current_reward:.0f} (obs:{obstacles_passed} + time:{steps*0.1:.0f})",
            f"Best Fitness: {self.best_fitness:.0f}",
        ]
        
        # Semi-transparent background for text
        overlay = pygame.Surface((420, 170))
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (10, 10))
        
        for i, text in enumerate(info_texts):
            surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (20, 20 + i * 30))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _tint_sprite(self, sprite, color):
        """Apply color tint to sprite"""
        tinted = sprite.copy()
        tinted.fill(color, special_flags=pygame.BLEND_MULT)
        return tinted
    
    def run(self, generations=100):
        """Run NEAT evolution for specified generations"""
        print(f"\n🧬 Starting NEAT Evolution for {generations} generations...")
        print(f"   Population: {self.config.pop_size}")
        print(f"   Inputs: 8 | Outputs: 3")
        print("="*60)
        
        # Run evolution
        winner = self.population.run(self.eval_genomes_parallel, generations)
        
        # Return the BEST OVERALL genome, not just last generation's winner
        if self.best_genome and self.best_genome.fitness > winner.fitness:
            winner = self.best_genome
            print(f"\n🏆 Evolution Complete!")
            print(f"   Using BEST EVER genome (fitness: {winner.fitness:.0f})")
        else:
            print(f"\n🏆 Evolution Complete!")
            print(f"   Best Fitness: {winner.fitness:.0f}")
        
        print(f"   Nodes: {len(winner.nodes)}")
        print(f"   Connections: {len([c for c in winner.connections.values() if c.enabled])}")
        
        return winner
    
    def save_winner(self, filepath, genome):
        """Save the winning genome"""
        with open(filepath, 'wb') as f:
            pickle.dump(genome, f)
        print(f"✓ Winner saved to {filepath}")
    
    def load_winner(self, filepath):
        """Load a saved genome"""
        with open(filepath, 'rb') as f:
            genome = pickle.load(f)
        return genome
    
    def save_checkpoint(self, filepath):
        """Save full population checkpoint for resuming training"""
        checkpoint = {
            'population': self.population,
            'generation': self.generation,
            'best_genome': self.best_genome,
            'best_fitness': self.best_fitness,
            'gen_best_scores': self.gen_best_scores,
            'gen_avg_scores': self.gen_avg_scores,
            'config': self.config
        }
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"✓ Checkpoint saved to {filepath}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, config_path, render=True):
        """Resume training from a saved checkpoint"""
        # Load checkpoint
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Create trainer instance
        trainer = cls.__new__(cls)
        trainer.config_path = config_path
        trainer.render = render
        trainer.generation = checkpoint['generation']
        trainer.best_genome = checkpoint['best_genome']
        trainer.best_fitness = checkpoint['best_fitness']
        trainer.gen_best_scores = checkpoint['gen_best_scores']
        trainer.gen_avg_scores = checkpoint['gen_avg_scores']
        trainer.config = checkpoint['config']
        trainer.population = checkpoint['population']
        
        # Re-add statistics reporter (no verbose stdout)
        trainer.stats = neat.StatisticsReporter()
        trainer.population.add_reporter(trainer.stats)
        
        # Setup pygame if rendering
        if render:
            pygame.init()
            trainer.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("🧬 NEAT Dino Evolution (Resumed)")
            trainer.clock = pygame.time.Clock()
            trainer.font = pygame.font.Font(None, 36)
            trainer.small_font = pygame.font.Font(None, 24)
            trainer.sprites = load_sprites()
            trainer.step_index = 0
            trainer.x_pos_bg = 0
        
        print(f"✓ Resumed from checkpoint: {checkpoint_path}")
        print(f"   Generation: {trainer.generation}")
        print(f"   Best Fitness: {trainer.best_fitness:.0f}")
        
        return trainer
    
    @classmethod
    def from_winner(cls, winner_path, config_path, render=True):
        """
        Resume training from a saved winner genome.
        Creates a new population seeded with mutations of the winner.
        """
        import copy
        
        # Create a normal trainer first
        trainer = cls(config_path, render=render)
        
        # Load the winner genome
        with open(winner_path, 'rb') as f:
            winner = pickle.load(f)
        
        # Store as best genome
        trainer.best_genome = winner
        trainer.best_fitness = winner.fitness if hasattr(winner, 'fitness') and winner.fitness else 0
        
        # Replace population with deep copies of the winner
        # Keep some exact copies and mutate the rest
        new_genomes = {}
        pop_size = trainer.config.pop_size
        
        for i in range(pop_size):
            # Deep copy the winner genome
            new_genome = copy.deepcopy(winner)
            new_genome.key = i  # Assign new key
            new_genome.fitness = None
            
            # Mutate half of them (keep 50% as exact copies)
            if i >= pop_size // 2:
                new_genome.mutate(trainer.config.genome_config)
            
            new_genomes[i] = new_genome
        
        # Replace population genomes
        trainer.population.population = new_genomes
        trainer.population.species.speciate(
            trainer.config, 
            trainer.population.population, 
            trainer.population.generation
        )
        
        print(f"✓ Loaded winner from: {winner_path}")
        print(f"   Winner Fitness: {trainer.best_fitness:.0f}")
        print(f"   Created {pop_size} genomes (50% copies, 50% mutations)")
        
        return trainer
    
    def close(self):
        """Clean up pygame"""
        if self.render:
            pygame.quit()


class NEATVisualTester:
    """Test and visualize a trained NEAT genome"""
    
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🦖 NEAT Dino - Testing Best Network")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def run(self, num_games=5):
        """Run test games with visualization"""
        scores = []
        
        for game_num in range(num_games):
            game = DinoGame()
            state = game.reset()
            
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return scores
                
                # Get action from network (2 outputs: JUMP, DUCK)
                output = self.net.activate(state)
                if output[0] > 0.5 and output[0] > output[1]:
                    action = 1  # JUMP
                elif output[1] > 0.5:
                    action = 2  # DUCK
                else:
                    action = 0  # RUN (default)
                
                # Step game
                state, _, done = game.step(action)
                
                # Render
                self.screen.fill((255, 255, 255))
                
                # Ground
                pygame.draw.line(self.screen, (0, 0, 0),
                               (0, GROUND_Y + 50), (SCREEN_WIDTH, GROUND_Y + 50), 2)
                
                # Dino
                current_height = game.duck_height if game.is_ducking else game.dino_height
                dino_color = (0, 200, 0) if not game.is_ducking else (0, 150, 0)
                pygame.draw.rect(self.screen, dino_color,
                               (80, game.dino_y - current_height + 50, 40, current_height))
                
                # Obstacles
                for obs in game.obstacles:
                    if obs['x'] > -100 and obs['x'] < SCREEN_WIDTH + 100:
                        color = (100, 100, 100) if obs['type'] == 'cactus' else (200, 50, 50)
                        pygame.draw.rect(self.screen, color,
                                       (obs['x'], obs['y'], obs['width'], obs['height']))
                
                # Info
                reward = (game.obstacles_passed * 10) + (game.steps * 0.1)
                texts = [
                    f"🦖 NEAT Test Game {game_num + 1}/{num_games}",
                    f"Score: {game.score:,}",
                    f"Reward: {reward:.0f} (obs:{game.obstacles_passed} + time:{game.steps*0.1:.0f})",
                    f"Action: {['RUN', 'JUMP', 'DUCK'][action]}"
                ]
                for i, text in enumerate(texts):
                    surface = self.font.render(text, True, (0, 0, 0))
                    self.screen.blit(surface, (20, 20 + i * 30))
                
                pygame.display.flip()
                self.clock.tick(60)
                
                if done:
                    scores.append(game.score)
                    print(f"  Game {game_num + 1}: Score = {game.score}")
                    pygame.time.wait(500)
                    running = False
        
        pygame.quit()
        return scores


def create_config_file(filepath, population_size=250):
    """Create NEAT configuration file with optimized settings for Dino game (NEAT 1.0 compatible)
    
    Args:
        filepath: Path to save the config file
        population_size: Number of genomes in the population (default 250)
    """
    config_content = f"""[NEAT]
fitness_criterion     = max
fitness_threshold     = 10000
pop_size              = {population_size}
reset_on_extinction   = True
no_fitness_termination = False

[DefaultGenome]
# Node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh relu sigmoid

# Node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum

# Node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_init_type          = gaussian
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# Genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.52

# Connection add/remove rates
conn_add_prob           = 0.3
conn_delete_prob        = 0.1

# Connection enable options
enabled_default              = True
enabled_mutate_rate          = 0.05
enabled_rate_to_true_add     = 0.0
enabled_rate_to_false_add    = 0.0

feed_forward            = True
initial_connection      = full_direct

# Node add/remove rates
node_add_prob           = 0.15
node_delete_prob        = 0.05

# Network parameters
num_hidden              = 0
num_inputs              = 12
num_outputs             = 2

# Node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_init_type      = gaussian
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# Connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_init_type        = gaussian
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.3
weight_mutate_rate      = 0.7
weight_replace_rate     = 0.1

# Structural mutation (NEAT 1.0 required)
single_structural_mutation = False
structural_mutation_surer  = default

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 5

[DefaultReproduction]
elitism            = 25
survival_threshold = 0.3
min_species_size   = 5
"""
    
    with open(filepath, 'w') as f:
        f.write(config_content)
    
    print(f"✓ NEAT config created: {filepath}")
    return filepath


if __name__ == "__main__":
    # Quick test
    config_path = "neat_config.txt"
    create_config_file(config_path)
    
    trainer = NEATTrainer(config_path, render=True)
    winner = trainer.run(generations=50)
    trainer.save_winner("neat_winner.pkl", winner)
    trainer.close()
