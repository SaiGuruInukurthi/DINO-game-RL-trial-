"""Multi-tab Dino game environment - Run multiple games in browser tabs

This runs N parallel Dino games in separate browser TABS (not windows).
Much more efficient than opening multiple browser windows!

Benefits:
- Single browser process
- Less memory usage
- Easier to manage
- Can train 10+ dinos simultaneously
"""
from __future__ import annotations

import time
import numpy as np
from typing import Tuple, Optional, Dict, List

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


class MultiTabDinoEnv:
    """
    Run multiple Dino games in parallel browser tabs.
    
    Each tab runs an independent game, all controlled from one browser.
    This is much more efficient than opening multiple browser windows!
    """
    
    def __init__(
        self,
        num_tabs: int = 10,
        game_url: str = "https://chromedino.com/",
        headless: bool = False
    ):
        self.num_tabs = num_tabs
        self.game_url = game_url
        self.headless = headless
        
        # State tracking per tab
        self.scores = [0] * num_tabs
        self.prev_scores = [0] * num_tabs
        self.is_game_over = [False] * num_tabs
        self.frames_alive = [0] * num_tabs
        
        # Observation/action spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=run, 1=jump, 2=duck
        
        # Initialize browser with tabs
        self.driver = None
        self.tab_handles = []
        self._init_browser()
        
        print(f"✅ Multi-tab environment ready with {num_tabs} parallel games!")
    
    def _init_browser(self):
        """Initialize Chrome browser and open game in multiple tabs"""
        print(f"🚀 Initializing browser with {self.num_tabs} tabs...")
        
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
        
        # Larger window to fit tabs
        chrome_options.add_argument('--window-size=1200,800')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        
        # Open first tab
        self.driver.get(self.game_url)
        time.sleep(2)
        self.tab_handles.append(self.driver.current_window_handle)
        print(f"  ✓ Tab 1/{self.num_tabs} ready")
        
        # Open additional tabs
        for i in range(1, self.num_tabs):
            # Open new tab with JavaScript
            self.driver.execute_script(f"window.open('{self.game_url}', '_blank');")
            time.sleep(0.5)
            
            # Get the new tab handle
            all_handles = self.driver.window_handles
            new_handle = [h for h in all_handles if h not in self.tab_handles][0]
            self.tab_handles.append(new_handle)
            
            print(f"  ✓ Tab {i+1}/{self.num_tabs} ready")
        
        # Inject state extractor into all tabs
        for i, handle in enumerate(self.tab_handles):
            self.driver.switch_to.window(handle)
            time.sleep(0.3)
            self._inject_state_extractor()
        
        print(f"✅ All {self.num_tabs} tabs initialized!")
    
    def _inject_state_extractor(self):
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
        self.driver.execute_script(js_code)
    
    def _switch_to_tab(self, tab_idx: int):
        """Switch to a specific tab"""
        self.driver.switch_to.window(self.tab_handles[tab_idx])
    
    def _get_game_state(self, tab_idx: int) -> Optional[Dict]:
        """Get game state from a specific tab"""
        self._switch_to_tab(tab_idx)
        try:
            return self.driver.execute_script("return window.getGameState();")
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
    
    def _start_game(self, tab_idx: int):
        """Start game in a specific tab"""
        self._switch_to_tab(tab_idx)
        body = self.driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.SPACE)
    
    def _send_action(self, tab_idx: int, action: int):
        """Send action to a specific tab"""
        self._switch_to_tab(tab_idx)
        body = self.driver.find_element(By.TAG_NAME, 'body')
        
        if action == 1:  # Jump
            body.send_keys(Keys.SPACE)
        elif action == 2:  # Duck
            body.send_keys(Keys.DOWN)
    
    def reset(self) -> Tuple[List[np.ndarray], List[dict]]:
        """Reset all games and return initial states"""
        states = []
        infos = []
        
        for i in range(self.num_tabs):
            self._switch_to_tab(i)
            
            # Start/restart game
            body = self.driver.find_element(By.TAG_NAME, 'body')
            body.send_keys(Keys.SPACE)
            
            # Reset state tracking
            self.scores[i] = 0
            self.prev_scores[i] = 0
            self.is_game_over[i] = False
            self.frames_alive[i] = 0
        
        time.sleep(0.5)  # Wait for games to start
        
        # Get initial states
        for i in range(self.num_tabs):
            game_state = self._get_game_state(i)
            obs = self._state_to_observation(game_state)
            states.append(obs)
            infos.append({'score': 0, 'tab': i})
        
        return states, infos
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[dict]]:
        """
        Execute actions in all tabs simultaneously.
        
        Args:
            actions: List of actions, one per tab
            
        Returns:
            states, rewards, terminateds, truncateds, infos
        """
        # Send actions to all tabs (fast loop)
        for i, action in enumerate(actions):
            if not self.is_game_over[i]:
                self._send_action(i, action)
        
        # Small delay for actions to take effect
        time.sleep(0.02)
        
        # Collect results from all tabs
        states = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i in range(self.num_tabs):
            self.frames_alive[i] += 1
            
            # Get game state
            game_state = self._get_game_state(i)
            
            # Check game over
            crashed = game_state.get('crashed', False) if game_state else False
            
            # Get score
            score = game_state.get('score', 0) if game_state else 0
            self.scores[i] = score
            
            # Calculate reward
            reward = 0.01  # Survival reward
            score_increase = self.scores[i] - self.prev_scores[i]
            if score_increase > 0:
                reward += score_increase * 1.0
            self.prev_scores[i] = self.scores[i]
            
            # Death penalty
            terminated = crashed
            if terminated:
                reward = -5.0
                self.is_game_over[i] = True
            
            # Get observation
            obs = self._state_to_observation(game_state)
            
            states.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(False)
            infos.append({
                'score': score,
                'tab': i,
                'frames': self.frames_alive[i]
            })
        
        return states, rewards, terminateds, truncateds, infos
    
    def reset_tab(self, tab_idx: int) -> Tuple[np.ndarray, dict]:
        """Reset a single tab (for when it crashes)"""
        self._switch_to_tab(tab_idx)
        body = self.driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.SPACE)
        
        self.scores[tab_idx] = 0
        self.prev_scores[tab_idx] = 0
        self.is_game_over[tab_idx] = False
        self.frames_alive[tab_idx] = 0
        
        time.sleep(0.3)
        
        game_state = self._get_game_state(tab_idx)
        obs = self._state_to_observation(game_state)
        
        return obs, {'score': 0, 'tab': tab_idx}
    
    def close(self):
        """Close browser"""
        if self.driver:
            try:
                self.driver.quit()
                print("✓ Browser closed")
            except:
                pass
    
    def __len__(self):
        return self.num_tabs


# Test
if __name__ == "__main__":
    print("Testing Multi-Tab Environment...\n")
    
    env = MultiTabDinoEnv(num_tabs=3)  # Test with 3 tabs
    
    print("\nResetting all games...")
    states, infos = env.reset()
    print(f"Got {len(states)} initial states")
    
    print("\nRunning 50 steps...")
    for step in range(50):
        # Random actions for all tabs
        actions = [np.random.randint(0, 3) for _ in range(env.num_tabs)]
        states, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Reset crashed tabs
        for i, done in enumerate(terminateds):
            if done:
                states[i], infos[i] = env.reset_tab(i)
                print(f"  Tab {i} crashed at step {step}, reset")
        
        if step % 10 == 0:
            scores = [info['score'] for info in infos]
            print(f"Step {step}: Scores = {scores}")
    
    env.close()
    print("\n✓ Test complete!")
