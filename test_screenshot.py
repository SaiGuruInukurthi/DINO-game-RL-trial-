"""Test screenshot capture and detection for Browser Dino Environment"""

import time
from src.browser_dino_env import BrowserDinoEnv
import numpy as np
from PIL import Image

def test_screenshot_capture():
    """Test if screenshots are being captured correctly"""
    print("🔍 Testing Screenshot Capture")
    print("=" * 60)
    
    # Initialize environment
    print("\n1. Initializing browser environment...")
    env = BrowserDinoEnv(headless=False, target_fps=20)
    
    try:
        # Reset and get initial screenshot
        print("\n2. Resetting environment and capturing initial screenshot...")
        obs, info = env.reset()
        
        print(f"   ✓ Screenshot captured!")
        print(f"   Shape: {obs.shape}")
        print(f"   Dtype: {obs.dtype}")
        print(f"   Min value: {obs.min()}")
        print(f"   Max value: {obs.max()}")
        print(f"   Mean value: {obs.mean():.2f}")
        
        # Check if screenshot is not blank
        if obs.max() == obs.min():
            print("   ⚠️  WARNING: Screenshot appears to be blank (all same value)")
        else:
            print("   ✓ Screenshot has variation (not blank)")
        
        # Save first screenshot
        img = Image.fromarray(obs, mode='L')
        img.save('screenshot_initial.png')
        print("   ✓ Saved as 'screenshot_initial.png'")
        
        # Capture multiple screenshots over time
        print("\n3. Capturing screenshots over 10 steps...")
        screenshots = []
        for i in range(10):
            action = 0  # Do nothing
            obs, reward, terminated, truncated, info = env.step(action)
            screenshots.append(obs.copy())
            
            if i % 3 == 0:
                print(f"   Step {i}: Mean={obs.mean():.2f}, Min={obs.min()}, Max={obs.max()}")
        
        # Check if screenshots are changing
        print("\n4. Checking if screenshots are changing...")
        differences = []
        for i in range(1, len(screenshots)):
            diff = np.abs(screenshots[i].astype(float) - screenshots[i-1].astype(float)).mean()
            differences.append(diff)
        
        avg_diff = np.mean(differences)
        print(f"   Average frame difference: {avg_diff:.2f}")
        
        if avg_diff < 1.0:
            print("   ⚠️  WARNING: Very little change between frames (possible issue)")
        else:
            print("   ✓ Frames are changing (good!)")
        
        # Save a few sample screenshots
        print("\n5. Saving sample screenshots...")
        for i in [0, 3, 6, 9]:
            img = Image.fromarray(screenshots[i], mode='L')
            img.save(f'screenshot_step_{i}.png')
        print("   ✓ Saved screenshots: screenshot_step_0.png, 3, 6, 9")
        
        # Test with a jump action
        print("\n6. Testing with jump action...")
        obs_before_jump = screenshots[-1]
        obs, reward, terminated, truncated, info = env.step(1)  # Jump
        time.sleep(0.5)  # Wait for jump animation
        obs_after_jump, reward, terminated, truncated, info = env.step(0)
        
        jump_diff = np.abs(obs_after_jump.astype(float) - obs_before_jump.astype(float)).mean()
        print(f"   Difference after jump: {jump_diff:.2f}")
        
        img = Image.fromarray(obs_after_jump, mode='L')
        img.save('screenshot_after_jump.png')
        print("   ✓ Saved 'screenshot_after_jump.png'")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"Screenshot shape: {obs.shape}")
        print(f"Average frame difference: {avg_diff:.2f}")
        print(f"Jump action difference: {jump_diff:.2f}")
        
        if obs.max() > obs.min() and avg_diff > 1.0:
            print("\n✅ Screenshot capture appears to be working!")
        else:
            print("\n⚠️  Screenshot capture may have issues")
            print("   Check the saved .png files to verify")
        
    finally:
        print("\n7. Closing browser...")
        env.close()
        print("✅ Test complete!")

if __name__ == "__main__":
    test_screenshot_capture()
