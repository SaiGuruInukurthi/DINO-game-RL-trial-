"""
NEAT Visual Trainer - See ALL dinos running together with REAL sprites!

Features:
- Uses original Chrome Dino Runner sprites
- Shows ALL population members running simultaneously  
- Each dino has unique color tint based on species
- Proper scoring (increments by 1, not 10s)
- Real physics from original game
"""

import neat
import pygame
import numpy as np
import os
import pickle
import random
from datetime import datetime

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 600

# Asset path
ASSET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Chrome-Dino-Runner", "assets")

# Load sprites
def load_sprites():
    """Load all game sprites"""
    sprites = {}
    
    # Dino sprites
    sprites['RUNNING'] = [
        pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoRun1.png")),
        pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoRun2.png")),
    ]
    sprites['JUMPING'] = pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoJump.png"))
    sprites['DUCKING'] = [
        pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoDuck1.png")),
        pygame.image.load(os.path.join(ASSET_PATH, "Dino", "DinoDuck2.png")),
    ]
    
    # Cactus sprites
    sprites['SMALL_CACTUS'] = [
        pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "SmallCactus1.png")),
        pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "SmallCactus2.png")),
        pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "SmallCactus3.png")),
    ]
    sprites['LARGE_CACTUS'] = [
        pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "LargeCactus1.png")),
        pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "LargeCactus2.png")),
        pygame.image.load(os.path.join(ASSET_PATH, "Cactus", "LargeCactus3.png")),
    ]
    
    # Bird sprites
    sprites['BIRD'] = [
        pygame.image.load(os.path.join(ASSET_PATH, "Bird", "Bird1.png")),
        pygame.image.load(os.path.join(ASSET_PATH, "Bird", "Bird2.png")),
    ]
    
    # Background
    sprites['BG'] = pygame.image.load(os.path.join(ASSET_PATH, "Other", "Track.png"))
    sprites['CLOUD'] = pygame.image.load(os.path.join(ASSET_PATH, "Other", "Cloud.png"))
    
    return sprites


def tint_sprite(sprite, color):
    """Apply a color tint to a sprite"""
    tinted = sprite.copy()
    tinted.fill(color, special_flags=pygame.BLEND_MULT)
    return tinted


class Dino:
    """Single dinosaur controlled by a NEAT network"""
    
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5
    
    def __init__(self, genome_id, genome, config, sprites, color):
        self.genome_id = genome_id
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.color = color
        self.sprites = sprites
        
        # Tint sprites for this dino
        self.run_img = [tint_sprite(s, color) for s in sprites['RUNNING']]
        self.jump_img = tint_sprite(sprites['JUMPING'], color)
        self.duck_img = [tint_sprite(s, color) for s in sprites['DUCKING']]
        
        # State
        self.alive = True
        self.score = 0
        self.fitness = 0
        
        # Physics (from original game)
        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False
        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        
        self.image = self.run_img[0]
        self.rect = self.image.get_rect()
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
    
    def think(self, obstacles, game_speed):
        """Use neural network to decide action"""
        if not self.alive or not obstacles:
            return
        
        # Get nearest obstacle
        nearest = None
        nearest_dist = float('inf')
        
        for obs in obstacles:
            dist = obs.rect.x - self.rect.x
            if dist > -50 and dist < nearest_dist:
                nearest = obs
                nearest_dist = dist
        
        if nearest is None:
            return
        
        # Build input state (normalized)
        inputs = [
            nearest_dist / SCREEN_WIDTH,                      # Distance to obstacle
            nearest.rect.width / 100.0,                       # Obstacle width
            nearest.rect.height / 100.0,                      # Obstacle height  
            (nearest.rect.y - 250) / 100.0,                   # Obstacle y position
            (self.rect.y - self.Y_POS) / 100.0,               # Dino y offset
            self.jump_vel / 10.0,                             # Jump velocity
            game_speed / 30.0,                                # Game speed
            1.0 if hasattr(nearest, 'is_bird') and nearest.is_bird else 0.0  # Is bird?
        ]
        
        # Get network output
        output = self.net.activate(inputs)
        
        # Decide action based on output
        action = np.argmax(output)
        
        if action == 1 and not self.dino_jump:  # JUMP
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif action == 2 and not self.dino_jump:  # DUCK
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif action == 0:  # RUN
            if not self.dino_jump:
                self.dino_duck = False
                self.dino_run = True
    
    def update(self):
        """Update dino physics"""
        if not self.alive:
            return
        
        if self.dino_duck:
            self.duck()
        elif self.dino_run:
            self.run()
        elif self.dino_jump:
            self.jump()
        
        if self.step_index >= 10:
            self.step_index = 0
    
    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.rect = self.image.get_rect()
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS_DUCK
        self.step_index += 1
    
    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.rect = self.image.get_rect()
        self.rect.x = self.X_POS
        self.rect.y = self.Y_POS
        self.step_index += 1
    
    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < -self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.rect.y = self.Y_POS
    
    def check_collision(self, obstacles):
        """Check collision with obstacles"""
        if not self.alive:
            return
        
        for obs in obstacles:
            if self.rect.colliderect(obs.rect):
                self.alive = False
                self.fitness = self.score
                self.genome.fitness = self.fitness
                return
    
    def draw(self, screen):
        """Draw dino if alive"""
        if self.alive:
            screen.blit(self.image, (self.rect.x, self.rect.y))


class Obstacle:
    """Base obstacle class"""
    def __init__(self, image, obs_type):
        self.image = image
        self.type = obs_type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH
        self.is_bird = False
    
    def update(self, game_speed):
        self.rect.x -= game_speed
    
    def draw(self, screen):
        screen.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, sprites):
        obs_type = random.randint(0, 2)
        super().__init__(sprites['SMALL_CACTUS'], obs_type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, sprites):
        obs_type = random.randint(0, 2)
        super().__init__(sprites['LARGE_CACTUS'], obs_type)
        self.rect.y = 300


class Bird(Obstacle):
    BIRD_HEIGHTS = [250, 290, 320]
    
    def __init__(self, sprites):
        super().__init__(sprites['BIRD'], 0)
        self.rect.y = random.choice(self.BIRD_HEIGHTS)
        self.is_bird = True
        self.index = 0
    
    def draw(self, screen):
        if self.index >= 9:
            self.index = 0
        screen.blit(self.image[self.index // 5], self.rect)
        self.index += 1


class NEATVisualTrainer:
    """
    NEAT trainer that shows ALL dinos running with real sprites!
    """
    
    def __init__(self, config_path, population_size=500):
        self.config_path = config_path
        self.population_size = population_size
        self.generation = 0
        self.best_score_ever = 0
        self.best_genome = None
        
        # Stats
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
        
        # Override population size
        self.config.pop_size = population_size
        
        # Create population
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        
        # Pygame setup
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("🧬 NEAT Dino Evolution - 500 Dinos!")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        
        # Load sprites
        self.sprites = load_sprites()
        
        # Generate species colors
        self.species_colors = self._generate_colors(50)
    
    def _generate_colors(self, n):
        """Generate n distinct colors for species"""
        colors = []
        for i in range(n):
            hue = i / n
            # HSV to RGB
            r, g, b = self._hsv_to_rgb(hue, 0.7, 1.0)
            colors.append((int(r*255), int(g*255), int(b*255)))
        return colors
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)
    
    def eval_genomes(self, genomes, config):
        """Evaluate all genomes by running them simultaneously"""
        self.generation += 1
        
        # Create dinos for each genome
        dinos = []
        for i, (genome_id, genome) in enumerate(genomes):
            genome.fitness = 0
            # Assign color based on index (species)
            color = self.species_colors[i % len(self.species_colors)]
            dino = Dino(genome_id, genome, config, self.sprites, color)
            dinos.append(dino)
        
        # Game state - use original NEAT speed (14)
        game_speed = 14
        score = 0
        obstacles = []
        x_pos_bg = 0
        y_pos_bg = 380
        
        # Run until all dinos are dead
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise KeyboardInterrupt("User closed window")
            
            # Check if any dinos alive
            alive_dinos = [d for d in dinos if d.alive]
            if not alive_dinos:
                break
            
            # Update score (proper increment by 1!)
            score += 1
            for dino in alive_dinos:
                dino.score = score
            
            # Increase speed every 100 points (cap at 25)
            if score % 100 == 0 and game_speed < 25:
                game_speed += 0.5
            
            # Spawn obstacles
            if len(obstacles) == 0 or obstacles[-1].rect.x < SCREEN_WIDTH - 300:
                r = random.randint(0, 2)
                if r == 0:
                    obstacles.append(SmallCactus(self.sprites))
                elif r == 1:
                    obstacles.append(LargeCactus(self.sprites))
                else:
                    obstacles.append(Bird(self.sprites))
            
            # Update obstacles
            for obs in obstacles:
                obs.update(game_speed)
            
            # Remove off-screen obstacles
            obstacles = [obs for obs in obstacles if obs.rect.x > -100]
            
            # Update dinos
            for dino in dinos:
                dino.think(obstacles, game_speed)
                dino.update()
                dino.check_collision(obstacles)
            
            # --- RENDER ---
            self.screen.fill((255, 255, 255))
            
            # Draw background
            image_width = self.sprites['BG'].get_width()
            self.screen.blit(self.sprites['BG'], (x_pos_bg, y_pos_bg))
            self.screen.blit(self.sprites['BG'], (image_width + x_pos_bg, y_pos_bg))
            if x_pos_bg <= -image_width:
                x_pos_bg = 0
            x_pos_bg -= game_speed
            
            # Draw obstacles
            for obs in obstacles:
                obs.draw(self.screen)
            
            # Draw ALL dinos (alive ones)
            for dino in alive_dinos:
                dino.draw(self.screen)
            
            # Draw stats
            alive_count = len(alive_dinos)
            stats_texts = [
                f"🧬 Generation: {self.generation}",
                f"🦖 Alive: {alive_count}/{len(dinos)}",
                f"📊 Score: {score}",
                f"🏆 Best Ever: {self.best_score_ever}",
                f"⚡ Speed: {game_speed}",
            ]
            
            for i, text in enumerate(stats_texts):
                surface = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(surface, (20, 20 + i * 25))
            
            # Draw species count
            species_text = f"Species: {len(self.population.species.species)}"
            surface = self.small_font.render(species_text, True, (100, 100, 100))
            self.screen.blit(surface, (SCREEN_WIDTH - 150, 20))
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        # Generation stats
        scores = [d.score for d in dinos]
        best_score = max(scores)
        avg_score = np.mean(scores)
        
        self.gen_best_scores.append(best_score)
        self.gen_avg_scores.append(avg_score)
        
        if best_score > self.best_score_ever:
            self.best_score_ever = best_score
            self.best_genome = max(dinos, key=lambda d: d.score).genome
        
        print(f"  Gen {self.generation}: Best={best_score}, Avg={avg_score:.0f}, Best Ever={self.best_score_ever}")
    
    def run(self, generations=100):
        """Run NEAT evolution"""
        print(f"\n🧬 Starting NEAT Evolution - {self.population_size} Dinos!")
        print(f"   Watch ALL dinos run together!")
        print("="*60)
        
        winner = self.population.run(self.eval_genomes, generations)
        
        return winner
    
    def save_winner(self, filepath, genome):
        """Save winning genome"""
        with open(filepath, 'wb') as f:
            pickle.dump(genome, f)
        print(f"✓ Winner saved: {filepath}")
    
    def close(self):
        """Clean up"""
        pygame.quit()


if __name__ == "__main__":
    from neat_trainer import create_config_file
    
    # Create config
    config_path = create_config_file("neat_config.txt")
    
    # Run trainer
    trainer = NEATVisualTrainer(config_path, population_size=100)
    try:
        winner = trainer.run(generations=50)
        trainer.save_winner("neat_winner.pkl", winner)
    finally:
        trainer.close()
