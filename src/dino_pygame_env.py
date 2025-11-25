"""
Pygame-based Dino Environment for RL Training.
Wraps the Chrome Dino Runner game (by dhhruv) as a Gymnasium environment.
Much faster than browser-based training!
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add Chrome-Dino-Runner to path
GAME_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Chrome-Dino-Runner')
sys.path.insert(0, GAME_DIR)
os.chdir(GAME_DIR)  # Change to game directory for asset loading

import pygame
import random


class DinoPygameEnv(gym.Env):
    """
    Gymnasium environment wrapper for Chrome Dino Runner.
    
    State (8 features):
        0: obstacle_distance (normalized 0-1)
        1: obstacle_height (normalized, 0=ground level)
        2: obstacle_width (normalized)
        3: dino_y_position (normalized, 0=ground)
        4: dino_is_jumping (0 or 1)
        5: game_speed (normalized)
        6: next_obstacle_distance (if exists, else 1.0)
        7: obstacle_type (0=cactus, 1=bird)
    
    Actions:
        0: Run (do nothing)
        1: Jump
        2: Duck
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    # Game constants
    SCREEN_WIDTH = 1100
    SCREEN_HEIGHT = 600
    DINO_X_POS = 80
    DINO_Y_POS = 310  # Ground level
    DINO_Y_POS_DUCK = 340
    JUMP_VEL = 8.5
    ACTION_COOLDOWN = 5  # Minimum frames between action changes (prevents spam)
    
    def __init__(self, render_mode='human', headless=False):
        super().__init__()
        
        self.render_mode = render_mode
        self.headless = headless
        
        # Action and observation space
        self.action_space = spaces.Discrete(3)  # 0=run, 1=jump, 2=duck
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Debug info (set by external trainer)
        self.debug_epsilon = 1.0
        self.debug_is_random = False
        self.last_action = 0
        
        # Initialize pygame
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Dino RL Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("freesansbold.ttf", 20)
        
        # Load assets
        self._load_assets()
        
        # Game state
        self.reset()
        
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
        self.CLOUD = pygame.image.load(os.path.join("assets/Other", "Cloud.png"))
        
    def reset(self, seed=None, options=None):
        """Reset the game to initial state"""
        super().reset(seed=seed)
        
        # Dino state
        self.dino_y = self.DINO_Y_POS
        self.dino_jump = False
        self.dino_duck = False
        self.dino_run = True
        self.jump_vel = self.JUMP_VEL
        self.step_index = 0
        
        # Game state
        self.game_speed = 8 # Start slower like original game
        self.score = 0
        self.obstacles = []
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.done = False
        
        # Cloud
        self.cloud_x = self.SCREEN_WIDTH + random.randint(800, 1000)
        self.cloud_y = random.randint(50, 100)
        
        # Spawn first obstacle
        self._spawn_obstacle()
        
        state = self._get_state()
        info = {'score': self.score, 'speed': self.game_speed}
        
        return state, info
    
    def _spawn_obstacle(self):
        """Spawn a new obstacle (cactus only for now - no birds)"""
        if len(self.obstacles) < 2:
            obstacle_type = random.randint(0, 1)  # Only cacti for now
            if obstacle_type == 0:
                cactus_type = random.randint(0, 2)
                img = self.SMALL_CACTUS[cactus_type]
                rect = img.get_rect()
                rect.x = self.SCREEN_WIDTH
                rect.y = 325
                self.obstacles.append({
                    'type': 'small_cactus',
                    'image': img,
                    'rect': rect,
                    'is_bird': False
                })
            else:
                cactus_type = random.randint(0, 2)
                img = self.LARGE_CACTUS[cactus_type]
                rect = img.get_rect()
                rect.x = self.SCREEN_WIDTH
                rect.y = 300
                self.obstacles.append({
                    'type': 'large_cactus',
                    'image': img,
                    'rect': rect,
                    'is_bird': False
                })
    
    def _get_state(self):
        """Extract state features for RL agent"""
        state = np.zeros(8, dtype=np.float32)
        
        # Get dino rect for collision
        if self.dino_jump:
            dino_img = self.JUMPING
        elif self.dino_duck:
            dino_img = self.DUCKING[0]
        else:
            dino_img = self.RUNNING[0]
        dino_rect = dino_img.get_rect()
        dino_rect.x = self.DINO_X_POS
        dino_rect.y = self.dino_y
        
        if self.obstacles:
            # Sort obstacles by distance
            sorted_obs = sorted(self.obstacles, key=lambda o: o['rect'].x)
            
            # Nearest obstacle
            nearest = sorted_obs[0]
            obs_distance = nearest['rect'].x - self.DINO_X_POS
            state[0] = np.clip(obs_distance / self.SCREEN_WIDTH, 0, 1)  # Distance
            state[1] = np.clip((self.DINO_Y_POS - nearest['rect'].y) / 100, 0, 1)  # Height diff
            state[2] = np.clip(nearest['rect'].width / 100, 0, 1)  # Width
            state[7] = 1.0 if nearest['is_bird'] else 0.0  # Type
            
            # Second obstacle (if exists)
            if len(sorted_obs) > 1:
                second = sorted_obs[1]
                state[6] = np.clip((second['rect'].x - self.DINO_X_POS) / self.SCREEN_WIDTH, 0, 1)
            else:
                state[6] = 1.0
        else:
            state[0] = 1.0  # No obstacle
            state[6] = 1.0
        
        # Dino state
        state[3] = np.clip((self.DINO_Y_POS - self.dino_y) / 150, 0, 1)  # Y position (0=ground)
        state[4] = 1.0 if self.dino_jump else 0.0  # Is jumping
        state[5] = np.clip(self.game_speed / 40, 0, 1)  # Speed normalized
        
        return state
    
    def step(self, action):
        """Execute one step in the environment"""
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
        
        # Action cooldown to prevent spam
        self.action_cooldown_timer = getattr(self, 'action_cooldown_timer', 0)
        self.last_action_taken = getattr(self, 'last_action_taken', 0)
        
        if self.action_cooldown_timer > 0:
            self.action_cooldown_timer -= 1
            # During cooldown, continue current action
            action = self.last_action_taken
        else:
            # New action allowed - reset cooldown if action changed
            if action != self.last_action_taken:
                self.action_cooldown_timer = self.ACTION_COOLDOWN
                self.last_action_taken = action
        
        # Apply action
        if action == 1 and not self.dino_jump:  # Jump
            self.dino_jump = True
            self.dino_run = False
            self.dino_duck = False
        elif action == 2 and not self.dino_jump:  # Duck (only if not jumping)
            self.dino_duck = True
            self.dino_run = False
        elif action == 0:  # Run
            if not self.dino_jump:  # Only change if not mid-jump
                self.dino_duck = False
                self.dino_run = True
        
        # Update dino
        self._update_dino()
        
        # Update obstacles
        self._update_obstacles()
        
        # Check collisions
        collision = self._check_collision()
        
        # Update score
        self.score += 1
        if self.score % 100 == 0 and self.game_speed < 20:  # Cap max speed at 15
            self.game_speed += 1
        
        # Calculate reward
        reward = self._calculate_reward(collision, action)
        
        # Check if done
        terminated = collision
        truncated = False
        
        # Render if needed
        if self.render_mode == 'human':
            self._render_frame()
        
        state = self._get_state()
        info = {'score': self.score, 'speed': self.game_speed}
        
        return state, reward, terminated, truncated, info
    
    def _update_dino(self):
        """Update dinosaur position and animation"""
        if self.dino_jump:
            self.dino_y -= self.jump_vel * 4
            self.jump_vel -= 0.8
            if self.jump_vel < -self.JUMP_VEL:
                self.dino_jump = False
                self.dino_run = True
                self.jump_vel = self.JUMP_VEL
                self.dino_y = self.DINO_Y_POS
        
        self.step_index += 1
        if self.step_index >= 10:
            self.step_index = 0
    
    def _update_obstacles(self):
        """Update obstacle positions and spawn new ones"""
        # Move obstacles
        for obs in self.obstacles[:]:
            obs['rect'].x -= self.game_speed
            if obs['rect'].x < -obs['rect'].width:
                self.obstacles.remove(obs)
        
        # Spawn new obstacle if needed
        if len(self.obstacles) == 0 or (
            len(self.obstacles) < 2 and 
            self.obstacles[-1]['rect'].x < self.SCREEN_WIDTH - 300
        ):
            self._spawn_obstacle()
        
        # Update background
        self.x_pos_bg -= self.game_speed
        if self.x_pos_bg <= -self.BG.get_width():
            self.x_pos_bg = 0
        
        # Update cloud
        self.cloud_x -= self.game_speed
        if self.cloud_x < -100:
            self.cloud_x = self.SCREEN_WIDTH + random.randint(800, 1000)
            self.cloud_y = random.randint(50, 100)
    
    def _check_collision(self):
        """Check if dino collides with any obstacle"""
        if self.dino_jump:
            dino_img = self.JUMPING
        elif self.dino_duck:
            dino_img = self.DUCKING[0]
        else:
            dino_img = self.RUNNING[0]
        
        dino_rect = dino_img.get_rect()
        dino_rect.x = self.DINO_X_POS
        dino_rect.y = self.dino_y
        
        # Shrink hitbox slightly for fairness
        dino_rect.inflate_ip(-10, -10)
        
        for obs in self.obstacles:
            obs_rect = obs['rect'].copy()
            obs_rect.inflate_ip(-5, -5)
            if dino_rect.colliderect(obs_rect):
                return True
        return False
    
    def _calculate_reward(self, collision, action):
        """Calculate reward for the current step"""
        if collision:
            return -100.0  # Death penalty
        
        reward = 0.1  # Small survival reward
        
        # Find nearest obstacle distance
        nearest_dist = 1000
        for obs in self.obstacles:
            dist = obs['rect'].x - self.DINO_X_POS
            if dist > 0:
                nearest_dist = min(nearest_dist, dist)
        
        # Bonus for passing obstacles
        for obs in self.obstacles:
            if obs['rect'].x + obs['rect'].width < self.DINO_X_POS:
                if not obs.get('passed', False):
                    obs['passed'] = True
                    reward += 15.0  # Obstacle passed bonus
        
        # Jump reward based on distance
        if action == 1:  # Tried to jump
            if nearest_dist > 50:
                reward -= 100.0  # Penalize jumping too early (far from obstacle)
            elif nearest_dist > 25:
                reward += 2.0  # Good jump distance
            else:
                reward += 0.5  # Okay but a bit late
        
        return reward
    
    def _render_frame(self):
        """Render the game to screen"""
        # Background
        self.screen.fill((255, 255, 255))
        
        # Track
        self.screen.blit(self.BG, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(self.BG, (self.x_pos_bg + self.BG.get_width(), self.y_pos_bg))
        
        # Cloud
        self.screen.blit(self.CLOUD, (self.cloud_x, self.cloud_y))
        
        # Dino
        if self.dino_jump:
            dino_img = self.JUMPING
        elif self.dino_duck:
            dino_img = self.DUCKING[self.step_index // 5]
        else:
            dino_img = self.RUNNING[self.step_index // 5]
        self.screen.blit(dino_img, (self.DINO_X_POS, self.dino_y))
        
        # Obstacles
        for obs in self.obstacles:
            self.screen.blit(obs['image'], obs['rect'])
        
        # Score and debug info
        text = self.font.render(f"Score: {self.score}  Speed: {self.game_speed}", True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        
        # Show epsilon and action source
        action_names = {0: "RUN", 1: "JUMP", 2: "DUCK"}
        action_str = action_names.get(self.last_action, "RUN")
        random_str = "RANDOM" if self.debug_is_random else "MODEL"
        color = (255, 0, 0) if self.debug_is_random else (0, 128, 0)
        debug_text = self.font.render(f"ε: {self.debug_epsilon:.3f}  Action: {action_str} ({random_str})", True, color)
        self.screen.blit(debug_text, (10, 35))
        
        # Show cooldown status
        cooldown = getattr(self, 'action_cooldown_timer', 0)
        if cooldown > 0:
            cd_text = self.font.render(f"Cooldown: {cooldown}", True, (128, 128, 128))
            self.screen.blit(cd_text, (10, 60))
        
        pygame.display.update()
        self.clock.tick(60)  # 30 FPS - normal game speed like original
    
    def render(self):
        """Render method for gymnasium compatibility"""
        if self.render_mode == 'human':
            self._render_frame()
        elif self.render_mode == 'rgb_array':
            return pygame.surfarray.array3d(self.screen)
    
    def close(self):
        """Clean up pygame"""
        pygame.quit()


class MultiDinoPygameEnv:
    """
    Multi-instance Pygame environment for parallel training.
    Runs multiple game instances in sequence (pygame is single-threaded).
    Still faster than browser-based due to no network overhead!
    """
    
    def __init__(self, num_envs=4, render_mode='human'):
        self.num_envs = num_envs
        self.render_mode = render_mode
        
        # Create single environment (pygame limitation)
        # We'll simulate multiple by running episodes in sequence
        print(f"🎮 Initializing Pygame Dino Environment")
        print(f"   Note: Pygame runs single-threaded, but it's still MUCH faster than browser!")
        
        self.env = DinoPygameEnv(render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self):
        """Reset the environment"""
        state, info = self.env.reset()
        return state, info
    
    def step(self, action):
        """Take a step in the environment"""
        return self.env.step(action)
    
    def close(self):
        """Close the environment"""
        self.env.close()


if __name__ == "__main__":
    # Test the environment
    env = DinoPygameEnv(render_mode='human')
    state, info = env.reset()
    print(f"Initial state: {state}")
    print(f"Info: {info}")
    
    # Run for a bit with random actions
    for i in range(500):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game over! Score: {info['score']}")
            state, info = env.reset()
    
    env.close()
