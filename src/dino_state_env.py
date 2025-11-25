"""State-based Dino game environment - Extract game data directly from browser

Instead of using screenshots (vision-based), this environment extracts
numerical game state from JavaScript (dino position, obstacle positions, speed, etc.)

This is MUCH faster and more reliable for RL training!
"""
from __future__ import annotations

import time
import numpy as np
from typing import Tuple, Optional, Dict, List
import json

try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
except ImportError:
    raise ImportError("Selenium required: pip install selenium")

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("Gymnasium required: pip install gymnasium")


class DinoStateEnv(gym.Env):
    """
    State-based Chrome Dino environment - NO screenshots, direct game state extraction.
    
    Observation Space (8 features):
        0. Dino Y position (0-100, normalized)
        1. Dino Y velocity (normalized)
        2. Game speed (normalized)
        3. Distance to nearest obstacle (0-1, normalized)
        4. Nearest obstacle width (normalized)
        5. Nearest obstacle height (normalized)
        6. Nearest obstacle type (0=cactus, 1=bird)
        7. Next obstacle distance (0-1, normalized)
        
    Action Space:
        - 0: Do nothing (run)
        - 1: Jump (press Space)
        - 2: Duck (press Down) - if needed for birds
        
    Rewards:
        - +0.01 per frame alive
        - +1.0 for each score point gained
        - -5.0 for death
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        headless: bool = False,
        game_url: str = "https://chromedino.com/",
        chromedriver_path: Optional[str] = None
    ):
        super().__init__()
        
        self.game_url = game_url
        self.headless = headless
        
        # Action space: 0=run, 1=jump, 2=duck
        self.action_space = spaces.Discrete(3)
        
        # Observation: 8 numerical features (much simpler than 80x80 image!)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # Initialize browser
        self.driver = None
        self.game_element = None
        self._init_browser(chromedriver_path)
        
        # Game state
        self.score = 0
        self.prev_score = 0
        self.is_game_over = False
        self.frames_alive = 0
        
        print("✓ State-based Dino environment initialized (no screenshots!)")
    
    def _init_browser(self, chromedriver_path: Optional[str] = None):
        """Initialize Chrome browser and load game"""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument('--headless')
        
        # Optimize for performance
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-logging')
        chrome_options.add_argument('--log-level=3')
        chrome_options.add_argument('--window-size=800,400')
        
        # Initialize driver
        if chromedriver_path:
            service = Service(chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            self.driver = webdriver.Chrome(options=chrome_options)
        
        # Load game
        self.driver.get(self.game_url)
        time.sleep(2)  # Wait for game to load
        
        # Get game canvas element
        self.game_element = self.driver.find_element(By.TAG_NAME, 'body')
        
        # Inject helper JavaScript to extract game state
        self._inject_state_extractor()
        
        print("✓ Browser initialized and state extractor injected")
    
    def _inject_state_extractor(self):
        """Inject JavaScript to extract game state data"""
        js_code = """
        // Create global function to extract game state
        window.getGameState = function() {
            try {
                // Access the game instance
                const runner = Runner.instance_;
                if (!runner) return null;
                
                // Get dino data
                const tRex = runner.tRex;
                const dinoY = tRex.yPos || 0;
                const dinoYVelocity = tRex.speedDrop ? -tRex.config.SPEED_DROP_COEFFICIENT : tRex.jumping ? -5 : 0;
                const ducking = tRex.ducking ? 1 : 0;
                
                // Get game speed
                const gameSpeed = runner.currentSpeed || 6;
                
                // Get obstacles
                const obstacles = runner.horizon.obstacles;
                const obstacleData = [];
                
                // Find up to 2 nearest obstacles ahead of dino
                for (let i = 0; i < obstacles.length && obstacleData.length < 2; i++) {
                    const obs = obstacles[i];
                    const distance = obs.xPos - tRex.xPos;
                    
                    if (distance > -20) {  // Only obstacles ahead or just passed
                        // Handle different obstacle types safely
                        const width = obs.width || obs.typeConfig?.width || 20;
                        const height = obs.height || obs.size?.height || obs.typeConfig?.height || 50;
                        const isPtero = obs.typeConfig?.type === 'PTERODACTYL' || obs.typeConfig?.type === 'pterodactyl';
                        
                        obstacleData.push({
                            distance: distance,
                            width: width,
                            height: height,
                            type: isPtero ? 1 : 0,
                            yPos: obs.yPos || 0
                        });
                    }
                }
                
                // Get score
                const score = runner.distanceMeter.getActualDistance(runner.distanceRan);
                
                // Check if game over
                const crashed = runner.crashed;
                
                return {
                    dinoY: dinoY,
                    dinoYVelocity: dinoYVelocity,
                    ducking: ducking,
                    gameSpeed: gameSpeed,
                    obstacles: obstacleData,
                    score: score,
                    crashed: crashed
                };
            } catch (e) {
                console.error("State extraction error:", e);
                return null;
            }
        };
        
        console.log("Game state extractor injected!");
        """
        
        self.driver.execute_script(js_code)
        time.sleep(0.5)
    
    def _get_game_state(self) -> Optional[Dict]:
        """Extract current game state from browser"""
        try:
            state_json = self.driver.execute_script("return window.getGameState();")
            
            # Debug: Print state on first call
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                if state_json:
                    print(f"\n🔍 Debug - Game state structure:")
                    print(f"   Keys: {list(state_json.keys())}")
                    print(f"   Sample: {state_json}")
                else:
                    print(f"\n⚠️  Game state is None - game may not be loaded yet")
            
            return state_json
        except Exception as e:
            print(f"Error extracting game state: {e}")
            return None
    
    def _state_to_observation(self, game_state: Dict) -> np.ndarray:
        """Convert raw game state to normalized observation vector"""
        if not game_state or game_state.get('crashed', False):
            # Return safe default state if crashed or error
            return np.zeros(8, dtype=np.float32)
        
        obs = np.zeros(8, dtype=np.float32)
        
        # Feature 0: Dino Y position (normalized to 0-1, typical range 0-100)
        obs[0] = np.clip(game_state.get('dinoY', 0) / 100.0, 0, 1)
        
        # Feature 1: Dino Y velocity (normalized, typical range -10 to +10)
        obs[1] = np.clip((game_state.get('dinoYVelocity', 0) + 10) / 20.0, 0, 1)
        
        # Feature 2: Game speed (normalized, typical range 6-13)
        obs[2] = np.clip((game_state.get('gameSpeed', 6) - 6) / 7.0, 0, 1)
        
        # Features 3-7: Obstacle data
        obstacles = game_state.get('obstacles', [])
        
        if len(obstacles) > 0:
            # Nearest obstacle
            nearest = obstacles[0]
            # Handle None/null values safely
            distance = nearest.get('distance', 600)
            width = nearest.get('width', 0)
            height = nearest.get('height', 0)
            obs_type = nearest.get('type', 0)
            
            # Normalize with safe defaults
            obs[3] = np.clip((distance if distance is not None else 600) / 600.0, 0, 1)
            obs[4] = np.clip((width if width is not None else 0) / 100.0, 0, 1)
            obs[5] = np.clip((height if height is not None else 0) / 100.0, 0, 1)
            obs[6] = float(obs_type if obs_type is not None else 0)
        else:
            # No obstacles visible
            obs[3] = 1.0  # Max distance
            obs[4] = 0.0
            obs[5] = 0.0
            obs[6] = 0.0
        
        # Feature 7: Next obstacle distance (if exists)
        if len(obstacles) > 1:
            next_dist = obstacles[1].get('distance', 600)
            obs[7] = np.clip((next_dist if next_dist is not None else 600) / 600.0, 0, 1)
        else:
            obs[7] = 1.0  # No second obstacle
        
        return obs
    
    def _start_game(self):
        """Start a new game by pressing Space"""
        self.game_element.send_keys(Keys.SPACE)
        time.sleep(0.3)
    
    def _is_game_over(self) -> bool:
        """Check if game is over via JavaScript"""
        try:
            return self.driver.execute_script(
                "return Runner.instance_ ? Runner.instance_.crashed : false;"
            )
        except:
            return False
    
    def _get_score(self) -> float:
        """Get current score via JavaScript"""
        try:
            game_state = self._get_game_state()
            if game_state:
                return game_state.get('score', 0)
            return 0
        except:
            return 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment and start new game"""
        super().reset(seed=seed)
        
        # Restart game by pressing Space
        self._start_game()
        time.sleep(0.5)
        
        # Reset state variables
        self.score = 0
        self.prev_score = 0
        self.is_game_over = False
        self.frames_alive = 0
        
        # Get initial state
        game_state = self._get_game_state()
        observation = self._state_to_observation(game_state)
        
        info = {
            'score': self.score,
            'frames': self.frames_alive
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=nothing, 1=jump, 2=duck
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute action
        if action == 1:  # Jump
            self.game_element.send_keys(Keys.SPACE)
        elif action == 2:  # Duck
            self.game_element.send_keys(Keys.DOWN)
        
        # Small delay for action to take effect
        time.sleep(0.03)  # Faster than screenshot version!
        
        # Update frame counter
        self.frames_alive += 1
        
        # Get game state
        game_state = self._get_game_state()
        
        # Check if game over
        self.is_game_over = game_state.get('crashed', False) if game_state else False
        
        # Get score
        self.score = game_state.get('score', 0) if game_state else 0
        
        # Calculate reward
        reward = 0.01  # Small survival reward per frame
        
        # Score-based reward (main objective)
        score_increase = self.score - self.prev_score
        if score_increase > 0:
            reward += score_increase * 1.0  # Strong reward for progress
        
        self.prev_score = self.score
        
        # Death penalty
        terminated = self.is_game_over
        if terminated:
            reward = -5.0
        
        truncated = False  # No truncation for now
        
        # Get observation from state
        observation = self._state_to_observation(game_state)
        
        info = {
            'score': self.score,
            'frames': self.frames_alive,
            'game_state': game_state
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render is handled by browser window"""
        pass
    
    def close(self):
        """Close the browser"""
        if self.driver:
            try:
                self.driver.quit()
                print("✓ Browser closed")
            except:
                pass


# Test function
if __name__ == "__main__":
    print("Testing State-based Dino Environment...\n")
    
    env = DinoStateEnv()
    
    print("\nObservation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    print("\nRunning test episode...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    for step in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: Score={info['score']:.0f}, Reward={reward:.2f}")
            print(f"  Obs: {obs}")
        
        if terminated:
            print(f"\nGame Over at step {step}!")
            print(f"Final score: {info['score']:.0f}")
            break
    
    env.close()
    print("\n✓ Test complete!")
