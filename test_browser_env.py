"""
Quick test script for browser environment

Run this on Windows (not WSL) after:
1. Installing Chrome browser
2. Downloading ChromeDriver
3. Installing dependencies: pip install -r requirements.txt
"""

from src.browser_dino_env import BrowserDinoEnv
import time

def test_basic():
    """Basic functionality test"""
    print("=" * 60)
    print("BROWSER DINO ENVIRONMENT TEST")
    print("=" * 60)
    print()
    
    print("Step 1: Initializing browser environment...")
    print("(This will open Chrome browser)")
    
    try:
        # Create environment (non-headless so you can see it)
        env = BrowserDinoEnv(
            headless=False,
            target_fps=20,
            chromedriver_path=None  # Will auto-detect
        )
        print("✓ Browser opened successfully!")
        print()
        
        print("Step 2: Resetting environment (starting game)...")
        obs, info = env.reset()
        print(f"✓ Game started!")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min()}, {obs.max()}]")
        print()
        
        print("Step 3: Running random actions for 50 steps...")
        print("Watch the browser - the agent is playing!")
        print()
        
        total_reward = 0
        for step in range(50):
            # Random action (0=nothing, 1=jump)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Print status every 10 steps
            if step % 10 == 0:
                print(f"  Step {step:3d}: Score={info['score']:4d}, "
                      f"FPS={info['fps']:5.1f}, Reward={reward:6.2f}")
            
            if terminated:
                print(f"\n  Game Over at step {step}!")
                print(f"  Final Score: {info['score']}")
                break
        
        print()
        print("Step 4: Cleanup...")
        env.close()
        print("✓ Browser closed")
        print()
        
        print("=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Summary:")
        print(f"  Steps completed: {step + 1}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final score: {info['score']}")
        print()
        print("Next steps:")
        print("  1. The environment is working!")
        print("  2. Create training notebook")
        print("  3. Implement DQN agent")
        print("  4. Start training!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Are you on Windows (not WSL)?")
        print("  2. Is Chrome browser installed?")
        print("  3. Is ChromeDriver downloaded and in PATH?")
        print("  4. Did you run: pip install -r requirements.txt")
        print()
        print("See BROWSER_SETUP.md for detailed instructions")
        raise

if __name__ == "__main__":
    test_basic()
