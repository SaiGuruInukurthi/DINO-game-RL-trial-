# Browser-Based Dino RL Setup Guide

## Prerequisites

1. **Google Chrome Browser** (latest version)
2. **ChromeDriver** matching your Chrome version
3. **Python 3.14** with DINO conda environment

## Step 1: Check Chrome Version

1. Open Chrome
2. Go to `chrome://settings/help`
3. Note your Chrome version (e.g., 120.0.6099.109)

## Step 2: Download ChromeDriver

1. Go to https://googlechromelabs.github.io/chrome-for-testing/
2. Download ChromeDriver for your Chrome version and OS (Windows)
3. Extract `chromedriver.exe`
4. Place it in one of these locations:
   - `C:\Program Files\chromedriver\chromedriver.exe`
   - OR in your project folder: `C:\Dino RL\chromedriver.exe`
   - OR add its folder to your system PATH

## Step 3: Install Python Dependencies

```bash
# Activate your conda environment
conda activate DINO

# Install all required packages
pip install -r requirements.txt
```

This will install:
- `selenium` - Browser automation
- `pillow` - Image processing
- `gymnasium` - RL environment interface
- `tensorflow` - Deep learning framework
- `numpy`, `matplotlib`, `opencv-python` - Utilities

## Step 4: Test the Browser Environment

```bash
# Test if everything works
python src/browser_dino_env.py
```

This should:
1. Open Chrome browser
2. Navigate to https://chromedino.com/
3. Play the game automatically for 100 steps
4. Print FPS and score information

## Step 5: Configure ChromeDriver Path (if needed)

If you get a "chromedriver not found" error, update the path:

```python
# In your code
from src.browser_dino_env import BrowserDinoEnv

env = BrowserDinoEnv(
    chromedriver_path="C:/path/to/chromedriver.exe",
    headless=False,  # Set True to hide browser window
    target_fps=20
)
```

## Troubleshooting

### Error: "ChromeDriver version mismatch"
- Download ChromeDriver matching your exact Chrome version

### Error: "selenium module not found"
```bash
pip install selenium
```

### Browser opens but game doesn't load
- Check internet connection
- Try different game URL: `game_url="https://chromedino.com/"`

### FPS is too slow (< 10 fps)
- Close other Chrome tabs
- Use `headless=True` mode
- Reduce screenshot processing overhead

### Multiple instances crash
- Ensure enough RAM (each instance uses ~200-500 MB)
- Use headless mode: `BrowserDinoEnv(headless=True)`
- Stagger startup (wait 5 sec between instances)

## Performance Tips for Training

1. **Headless Mode**: 20-30% faster
   ```python
   env = BrowserDinoEnv(headless=True)
   ```

2. **Target FPS**: Set realistic based on your hardware
   ```python
   env = BrowserDinoEnv(target_fps=20)  # i5-12450H + RTX 3050 Mobile
   ```

3. **Parallel Training**: Run 2-3 instances
   ```python
   # In separate processes/terminals
   env1 = BrowserDinoEnv(headless=True, target_fps=20)
   env2 = BrowserDinoEnv(headless=True, target_fps=20)
   ```

## Next Steps

Once the environment is working:
1. Test with random actions
2. Create training notebook
3. Implement DQN agent
4. Train and evaluate!

Good luck! 🦖🎮
