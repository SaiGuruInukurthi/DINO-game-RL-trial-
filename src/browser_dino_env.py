"""Browser-based Dino game environment using Selenium

This module provides a Gymnasium-compatible environment that controls
the actual Chrome Dino game in a browser using Selenium WebDriver.

Target URL: https://chromedino.com/
"""
from __future__ import annotations

import time
import numpy as np
from typing import Tuple, Optional
from PIL import Image
import io

try:
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
except ImportError:
    raise ImportError(
        "Selenium is required for browser environment. "
        "Install with: pip install selenium"
    )

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "Gymnasium is required. Install with: pip install gymnasium"
    )


class BrowserDinoEnv(gym.Env):
    """
    Gymnasium environment for Chrome Dino game via browser automation.
    
    Observation Space:
        - Screenshot-based: Grayscale image of game area (150x600 pixels)
        - Preprocessed to 80x80 for neural network input
        
    Action Space:
        - 0: Do nothing (run)
        - 1: Jump (press Space)
        - 2: Duck (press Down arrow) - if supported
        
    Rewards:
        - +0.1 for each frame alive
        - -10 for game over
        - Bonus based on score increase
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        headless: bool = False,
        game_url: str = "https://chromedino.com/",
        chromedriver_path: Optional[str] = None,
        target_fps: int = 20
    ):
        """
        Initialize the browser-based Dino environment.
        
        Args:
            headless: Run browser in headless mode (no GUI)
            game_url: URL of the Chrome Dino game
            chromedriver_path: Path to chromedriver executable (None = auto-detect)
            target_fps: Target frames per second for game loop
        """
        super().__init__()
        
        self.game_url = game_url
        self.headless = headless
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0=nothing, 1=jump (duck disabled for now)
        
        # Observation: 80x80 grayscale image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(80, 80),
            dtype=np.uint8
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
        
    def _init_browser(self, chromedriver_path: Optional[str] = None):
        """Initialize Chrome browser with Selenium"""
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
        
        # Block ads and unnecessary content
        chrome_options.add_argument('--disable-popup-blocking')
        chrome_options.add_experimental_option('prefs', {
            'profile.default_content_setting_values.notifications': 2,
            'profile.default_content_setting_values.ads': 2
        })
        
        # Initialize driver
        if chromedriver_path:
            service = Service(chromedriver_path)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            # Try to use chromedriver from PATH
            self.driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to game
        self.driver.get(self.game_url)
        time.sleep(2)  # Wait for page load
        
        # Hide ads and unnecessary elements via JavaScript
        try:
            self.driver.execute_script("""
                // Hide ads and unnecessary content
                var ads = document.querySelectorAll('iframe, .ad, .advertisement, [id*="ad"], [class*="ad"]');
                ads.forEach(function(ad) { 
                    if (ad.tagName !== 'CANVAS') {
                        ad.style.display = 'none'; 
                    }
                });
                
                // Hide everything below the canvas
                var canvas = document.querySelector('canvas');
                if (canvas) {
                    var parent = canvas.parentElement;
                    var siblings = Array.from(parent.parentElement.children);
                    var canvasIndex = siblings.indexOf(parent);
                    for (var i = canvasIndex + 1; i < siblings.length; i++) {
                        siblings[i].style.display = 'none';
                    }
                }
                
                // Set page background to black for better contrast
                document.body.style.backgroundColor = '#000000';
            """)
            print("✓ Ads and unnecessary content hidden")
        except Exception as e:
            print(f"Warning: Could not hide ads: {e}")
        
        # Find game canvas/element
        try:
            # Try to find the game canvas or container
            self.game_element = self.driver.find_element(By.TAG_NAME, 'body')
        except Exception as e:
            print(f"Warning: Could not locate game element: {e}")
            self.game_element = self.driver.find_element(By.TAG_NAME, 'body')
    
    def _start_game(self):
        """Start the game by pressing Space"""
        self.game_element.send_keys(Keys.SPACE)
        time.sleep(0.1)
    
    def _get_screenshot(self, debug=False) -> np.ndarray:
        """
        Capture game screenshot and preprocess it.
        
        Args:
            debug: If True, save debug images showing crop regions
        
        Returns:
            80x80 grayscale numpy array
        """
        try:
            # Try to get canvas screenshot first (most accurate - pure game, no extras)
            canvas = self.driver.find_element(By.TAG_NAME, 'canvas')
            png = canvas.screenshot_as_png
            img = Image.open(io.BytesIO(png))
            
            if debug:
                img.save('debug_canvas.png')
                print(f"Debug: Canvas captured - size = {img.size}")
            
            # Canvas is pure game content - use it directly
            game_region = img.convert('L')
            
        except Exception as e:
            # Fallback: This should rarely happen, but if canvas capture fails,
            # get full page and crop to just the game area (top portion only)
            if debug:
                print(f"Canvas capture failed: {e}, using full page with precise crop")
            
            png = self.driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(png))
            
            if debug:
                img.save('debug_fullpage.png')
                print(f"Debug: Full page size = {img.size}")
            
            # Convert to grayscale first
            img_gray = img.convert('L')
            
            # Crop to ONLY the game canvas area (top ~150-200px of the page)
            # The game is at the top, everything else (scores, buttons) is below
            width, height = img_gray.size
            
            # Crop to just the top portion where the game canvas is
            # Typically the canvas is in the top 200-300px
            left = 0
            top = 0
            right = width
            bottom = min(int(height * 0.35), 300)  # Only top 35% or 300px max
            
            game_region = img_gray.crop((left, top, right, bottom))
            
            if debug:
                img_gray.save('debug_grayscale.png')
                game_region.save('debug_cropped.png')
                print(f"Debug: Cropped to {game_region.size} from top portion only")
        
        # Resize to 80x80 for neural network
        game_region_resized = game_region.resize((80, 80), Image.Resampling.LANCZOS)
        
        if debug:
            game_region_resized.save('debug_final_80x80.png')
            print(f"Debug: Final size = {game_region_resized.size}")
        
        # Convert to numpy array
        img_array = np.array(game_region_resized, dtype=np.uint8)
        
        return img_array
    
    def _get_score(self) -> int:
        """
        Extract current score from the game using JavaScript API.
        
        Returns:
            Current game score (distance traveled)
        """
        try:
            # Access the game's internal distance counter
            # distanceRan is the actual distance traveled (score)
            score = self.driver.execute_script(
                "return Runner && Runner.instance_ ? "
                "Math.floor(Runner.instance_.distanceRan) : 0"
            )
            return int(score) if score else 0
        except Exception as e:
            # Fallback: try to read score from canvas if available
            try:
                score_text = self.driver.execute_script(
                    "return Runner && Runner.instance_ ? "
                    "Runner.instance_.distanceMeter.getActualDistance(0) : 0"
                )
                return int(score_text) if score_text else 0
            except:
                return 0
    
    def _is_game_over(self) -> bool:
        """
        Check if game is over using JavaScript state detection.
        
        Returns:
            True if game is over (dinosaur crashed)
        """
        try:
            # Primary: Check the game's internal crashed state
            is_crashed = self.driver.execute_script(
                "return Runner && Runner.instance_ ? "
                "Runner.instance_.crashed : false"
            )
            
            if is_crashed:
                return True
            
            # Secondary: Check if game is paused (another indicator)
            is_paused = self.driver.execute_script(
                "return Runner && Runner.instance_ ? "
                "Runner.instance_.paused : false"
            )
            
            # Game is over if crashed or paused after starting
            return bool(is_crashed or (is_paused and self.frames_alive > 10))
            
        except Exception as e:
            # Fallback: Look for restart button visibility
            try:
                restart_visible = self.driver.execute_script(
                    "var btn = document.querySelector('.icon-restart'); "
                    "return btn ? btn.offsetWidth > 0 : false"
                )
                return bool(restart_visible)
            except:
                return False
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Refresh page to restart game
        self.driver.refresh()
        time.sleep(2)
        
        # Re-hide ads after refresh
        try:
            self.driver.execute_script("""
                var ads = document.querySelectorAll('iframe, .ad, .advertisement, [id*="ad"], [class*="ad"]');
                ads.forEach(function(ad) { 
                    if (ad.tagName !== 'CANVAS') {
                        ad.style.display = 'none'; 
                    }
                });
                
                var canvas = document.querySelector('canvas');
                if (canvas) {
                    var parent = canvas.parentElement;
                    var siblings = Array.from(parent.parentElement.children);
                    var canvasIndex = siblings.indexOf(parent);
                    for (var i = canvasIndex + 1; i < siblings.length; i++) {
                        siblings[i].style.display = 'none';
                    }
                }
                document.body.style.backgroundColor = '#000000';
            """)
        except:
            pass
        
        # Re-find game element after refresh (prevents stale element error)
        self.game_element = self.driver.find_element(By.TAG_NAME, 'body')
        
        # Start game
        self._start_game()
        time.sleep(0.5)
        
        # Reset state
        self.score = 0
        self.prev_score = 0
        self.is_game_over = False
        self.frames_alive = 0
        
        # Get initial observation
        obs = self._get_screenshot()
        
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=nothing, 1=jump
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        start_time = time.time()
        
        # Execute action
        if action == 1:  # Jump
            self.game_element.send_keys(Keys.SPACE)
        # Note: Duck (action 2) would use Keys.DOWN if needed
        
        # Small delay to let action take effect
        time.sleep(0.05)
        
        # Update frame counter
        self.frames_alive += 1
        
        # Get new state
        obs = self._get_screenshot()
        self.score = self._get_score()
        self.is_game_over = self._is_game_over()
        
        # Calculate reward
        reward = 0.1  # Small reward for staying alive
        
        # Score-based reward
        score_increase = self.score - self.prev_score
        if score_increase > 0:
            reward += score_increase * 0.1
        
        self.prev_score = self.score
        
        # Game over penalty
        terminated = self.is_game_over
        if terminated:
            reward = -10.0
        
        truncated = False
        info = {
            'score': self.score,
            'frames': self.frames_alive,
            'fps': 1.0 / max(time.time() - start_time, 0.001)
        }
        
        # Frame rate limiting
        elapsed = time.time() - start_time
        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render is handled by the browser window"""
        pass
    
    def close(self):
        """Clean up browser resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None


# Quick test function
def test_browser_env():
    """Test the browser environment"""
    print("Initializing browser environment...")
    env = BrowserDinoEnv(headless=False, target_fps=20)
    
    try:
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        
        print("Running random actions for 100 steps...")
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i % 20 == 0:
                print(f"Step {i}: Score={info['score']}, FPS={info['fps']:.1f}, Reward={reward:.2f}")
            
            if terminated:
                print(f"Game over at step {i}! Final score: {info['score']}")
                obs, info = env.reset()
                
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    test_browser_env()
