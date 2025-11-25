"""Multi-window Dino game environment - Run multiple games in separate browser windows

This runs N parallel Dino games in separate browser WINDOWS.
Each window runs independently, allowing true parallel training.

Benefits:
- True parallel execution
- Each window is independent
- Can use multiple CPU cores for browser operations
- Works well with GPU training (single model, multiple data sources)
"""
from __future__ import annotations

import time
import numpy as np
from typing import Tuple, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
except ImportError:
    raise ImportError("Selenium required: pip install selenium")

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("Gymnasium required: pip install gymnasium")


class MultiWindowDinoEnv:
    """
    Run multiple Dino games in parallel browser windows.
    
    Each window runs an independent game with its own browser instance.
    This allows true parallel data collection while training on GPU.
    """
    
    def __init__(
        self,
        num_windows: int = 4,
        game_url: str = "https://chromedino.com/",
        headless: bool = False
    ):
        self.num_windows = num_windows
        self.game_url = game_url
        self.headless = headless
        
        # State tracking per window
        self.scores = [0] * num_windows
        self.prev_scores = [0] * num_windows
        self.is_game_over = [False] * num_windows
        self.frames_alive = [0] * num_windows
        
        # Observation/action spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=run, 1=jump, 2=duck
        
        # Browser instances (one per window)
        self.drivers: List[webdriver.Chrome] = []
        self.locks = [threading.Lock() for _ in range(num_windows)]
        
        # Initialize browsers
        self._init_browsers()
        
        print(f"✅ Multi-window environment ready with {num_windows} parallel games!")
    
    def _init_browsers(self):
        """Initialize Chrome browsers in separate windows"""
        print(f"🚀 Initializing {self.num_windows} browser windows...")
        
        # Calculate window positions for a grid layout
        screen_width = 1920
        screen_height = 1080
        
        if self.num_windows <= 2:
            cols, rows = self.num_windows, 1
        elif self.num_windows <= 4:
            cols, rows = 2, 2
        elif self.num_windows <= 6:
            cols, rows = 3, 2
        else:
            cols, rows = 4, 2
        
        win_width = screen_width // cols
        win_height = screen_height // rows
        
        for i in range(self.num_windows):
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless')
            
            # Performance optimizations
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-logging')
            chrome_options.add_argument('--log-level=3')
            chrome_options.add_argument('--mute-audio')
            
            # Window size
            chrome_options.add_argument(f'--window-size={win_width},{win_height}')
            
            # Calculate position
            col = i % cols
            row = i // cols
            x_pos = col * win_width
            y_pos = row * win_height
            chrome_options.add_argument(f'--window-position={x_pos},{y_pos}')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.game_url)
            self.drivers.append(driver)
            
            print(f"  ✓ Window {i+1}/{self.num_windows} ready at ({x_pos}, {y_pos})")
        
        time.sleep(2)  # Wait for all pages to load
        
        # Inject state extractor into all windows
        for i, driver in enumerate(self.drivers):
            self._inject_state_extractor(driver)
        
        print(f"✅ All {self.num_windows} windows initialized!")
    
    def _inject_state_extractor(self, driver):
        """Inject JavaScript to extract game state"""
        js_code = """
        window.getGameState = function() {
            try {
                const runner = Runner.instance_;
                if (!runner) return null;
                
                const tRex = runner.tRex;
                const dinoY = tRex.yPos || 0;
                const dinoYVelocity = tRex.speedDrop ? -tRex.config.SPEED_DROP_COEFFICIENT : tRex.jumping ? -5 : 0;
                const gameSpeed = runner.currentSpeed || 6;
                
                const obstacles = runner.horizon.obstacles;
                const obstacleData = [];
                
                for (let i = 0; i < obstacles.length && obstacleData.length < 2; i++) {
                    const obs = obstacles[i];
                    const distance = obs.xPos - tRex.xPos;
                    
                    if (distance > -20) {
                        const width = obs.width || obs.typeConfig?.width || 20;
                        const height = obs.height || obs.size?.height || obs.typeConfig?.height || 50;
                        const isPtero = obs.typeConfig?.type === 'PTERODACTYL';
                        
                        obstacleData.push({
                            distance: distance,
                            width: width,
                            height: height,
                            type: isPtero ? 1 : 0,
                            yPos: obs.yPos || 0
                        });
                    }
                }
                
                const score = runner.distanceMeter.getActualDistance(runner.distanceRan);
                const crashed = runner.crashed;
                
                return {
                    dinoY: dinoY,
                    dinoYVelocity: dinoYVelocity,
                    gameSpeed: gameSpeed,
                    obstacles: obstacleData,
                    score: score,
                    crashed: crashed
                };
            } catch (e) {
                return null;
            }
        };
        """
        driver.execute_script(js_code)
    
    def _get_game_state(self, window_idx: int) -> Optional[Dict]:
        """Get game state from a specific window"""
        try:
            return self.drivers[window_idx].execute_script("return window.getGameState();")
        except:
            return None
    
    def _state_to_observation(self, game_state: Dict) -> np.ndarray:
        """Convert game state to observation vector"""
        if not game_state:
            return np.zeros(8, dtype=np.float32)
        
        obs = np.zeros(8, dtype=np.float32)
        
        obs[0] = np.clip((game_state.get('dinoY', 0) or 0) / 100.0, 0, 1)
        obs[1] = np.clip(((game_state.get('dinoYVelocity', 0) or 0) + 10) / 20.0, 0, 1)
        obs[2] = np.clip(((game_state.get('gameSpeed', 6) or 6) - 6) / 7.0, 0, 1)
        
        obstacles = game_state.get('obstacles', [])
        if len(obstacles) > 0:
            nearest = obstacles[0]
            obs[3] = np.clip((nearest.get('distance', 600) or 600) / 600.0, 0, 1)
            obs[4] = np.clip((nearest.get('width', 0) or 0) / 100.0, 0, 1)
            obs[5] = np.clip((nearest.get('height', 0) or 0) / 100.0, 0, 1)
            obs[6] = float(nearest.get('type', 0) or 0)
        else:
            obs[3] = 1.0
        
        if len(obstacles) > 1:
            obs[7] = np.clip((obstacles[1].get('distance', 600) or 600) / 600.0, 0, 1)
        else:
            obs[7] = 1.0
        
        return obs
    
    def _send_action(self, window_idx: int, action: int):
        """Send action to a specific window"""
        try:
            body = self.drivers[window_idx].find_element(By.TAG_NAME, 'body')
            if action == 1:  # Jump
                body.send_keys(Keys.SPACE)
            elif action == 2:  # Duck
                body.send_keys(Keys.DOWN)
        except:
            pass
    
    def _process_window(self, window_idx: int, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Process a single window - get state, send action, calculate reward"""
        with self.locks[window_idx]:
            # Send action first
            if not self.is_game_over[window_idx]:
                self._send_action(window_idx, action)
            
            self.frames_alive[window_idx] += 1
            
            # Get game state
            game_state = self._get_game_state(window_idx)
            
            # Check game over
            crashed = game_state.get('crashed', False) if game_state else False
            
            # Get score
            score = game_state.get('score', 0) if game_state else 0
            self.scores[window_idx] = score
            
            # Calculate reward
            reward = 0.01  # Small survival reward
            score_increase = self.scores[window_idx] - self.prev_scores[window_idx]
            if score_increase > 0:
                reward += score_increase * 1.0
            self.prev_scores[window_idx] = self.scores[window_idx]
            
            # Death penalty
            terminated = crashed
            if terminated:
                reward = -5.0
                self.is_game_over[window_idx] = True
            
            # Get observation
            obs = self._state_to_observation(game_state)
            
            info = {
                'score': score,
                'window': window_idx,
                'frames': self.frames_alive[window_idx]
            }
            
            return obs, reward, terminated, info
    
    def reset(self) -> Tuple[List[np.ndarray], List[dict]]:
        """Reset all games and return initial states"""
        states = []
        infos = []
        
        for i in range(self.num_windows):
            # Start/restart game
            try:
                body = self.drivers[i].find_element(By.TAG_NAME, 'body')
                body.send_keys(Keys.SPACE)
            except:
                pass
            
            # Reset state tracking
            self.scores[i] = 0
            self.prev_scores[i] = 0
            self.is_game_over[i] = False
            self.frames_alive[i] = 0
        
        time.sleep(0.5)  # Wait for games to start
        
        # Get initial states
        for i in range(self.num_windows):
            game_state = self._get_game_state(i)
            obs = self._state_to_observation(game_state)
            states.append(obs)
            infos.append({'score': 0, 'window': i})
        
        return states, infos
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict]]:
        """
        Execute actions in all windows using thread pool.
        
        Args:
            actions: List of actions, one per window
            
        Returns:
            states, rewards, terminateds, truncateds, infos
        """
        states = [None] * self.num_windows
        rewards = [0.0] * self.num_windows
        terminateds = [False] * self.num_windows
        truncateds = [False] * self.num_windows
        infos = [{}] * self.num_windows
        
        # Process all windows in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_windows) as executor:
            futures = {
                executor.submit(self._process_window, i, actions[i]): i 
                for i in range(self.num_windows)
            }
            
            for future in as_completed(futures):
                i = futures[future]
                try:
                    obs, reward, terminated, info = future.result()
                    states[i] = obs
                    rewards[i] = reward
                    terminateds[i] = terminated
                    infos[i] = info
                except Exception as e:
                    # Fallback for failed window
                    states[i] = np.zeros(8, dtype=np.float32)
                    rewards[i] = -5.0
                    terminateds[i] = True
                    infos[i] = {'score': 0, 'window': i, 'error': str(e)}
        
        return states, rewards, terminateds, truncateds, infos
    
    def reset_window(self, window_idx: int) -> Tuple[np.ndarray, dict]:
        """Reset a single window (for when it crashes)"""
        with self.locks[window_idx]:
            try:
                body = self.drivers[window_idx].find_element(By.TAG_NAME, 'body')
                body.send_keys(Keys.SPACE)
            except:
                pass
            
            self.scores[window_idx] = 0
            self.prev_scores[window_idx] = 0
            self.is_game_over[window_idx] = False
            self.frames_alive[window_idx] = 0
        
        time.sleep(0.3)
        
        game_state = self._get_game_state(window_idx)
        obs = self._state_to_observation(game_state)
        
        return obs, {'score': 0, 'window': window_idx}
    
    def close(self):
        """Close all browser windows"""
        for i, driver in enumerate(self.drivers):
            try:
                driver.quit()
                print(f"  ✓ Window {i+1} closed")
            except:
                pass
        self.drivers = []
        print("✓ All browsers closed")
    
    def __len__(self):
        return self.num_windows


# Test
if __name__ == "__main__":
    print("Testing Multi-Window Environment...\n")
    
    env = MultiWindowDinoEnv(num_windows=2)  # Test with 2 windows
    
    print("\nResetting all games...")
    states, infos = env.reset()
    print(f"Got {len(states)} initial states")
    
    print("\nRunning 50 steps...")
    for step in range(50):
        # Random actions for all windows
        actions = [np.random.randint(0, 3) for _ in range(env.num_windows)]
        states, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Reset crashed windows
        for i, done in enumerate(terminateds):
            if done:
                states[i], infos[i] = env.reset_window(i)
                print(f"  Window {i} crashed at step {step}, reset")
        
        if step % 10 == 0:
            scores = [info['score'] for info in infos]
            print(f"Step {step}: Scores = {scores}")
    
    env.close()
    print("\n✓ Test complete!")
