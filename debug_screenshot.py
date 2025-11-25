"""Debug screenshot capture to see what we're actually capturing"""

from src.browser_dino_env import BrowserDinoEnv
import time

def debug_screenshot():
    print("🔍 Debug Screenshot Capture")
    print("=" * 60)
    
    env = BrowserDinoEnv(headless=False, target_fps=20)
    
    try:
        print("\n1. Resetting environment...")
        obs, info = env.reset()
        
        print("\n2. Waiting 2 seconds for game to start...")
        time.sleep(2)
        
        print("\n3. Capturing DEBUG screenshot...")
        # Call with debug=True
        obs = env._get_screenshot(debug=True)
        
        print(f"\n4. Screenshot stats:")
        print(f"   Shape: {obs.shape}")
        print(f"   Min: {obs.min()}, Max: {obs.max()}, Mean: {obs.mean():.2f}")
        
        print("\n5. Taking an action (jump) and capturing again...")
        env.step(1)  # Jump
        time.sleep(0.3)
        obs = env._get_screenshot(debug=True)
        
        print("\n✅ Check the debug_*.png files to see what's being captured!")
        print("   - debug_canvas.png or debug_fullpage.png: Raw capture")
        print("   - debug_grayscale.png: After grayscale conversion")
        print("   - debug_cropped.png: After cropping")
        print("   - debug_final_80x80.png: Final processed image")
        
    finally:
        input("\nPress Enter to close browser...")
        env.close()

if __name__ == "__main__":
    debug_screenshot()
