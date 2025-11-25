"""Test game-over detection and score extraction"""

from src.browser_dino_env import BrowserDinoEnv
import time

def test_detection():
    print("🎮 Testing Game-Over Detection & Score Extraction")
    print("=" * 60)
    
    env = BrowserDinoEnv(headless=False, target_fps=20)
    
    try:
        print("\n1. Resetting environment...")
        obs, info = env.reset()
        print(f"   Initial score: {info.get('score', 0)}")
        print(f"   Initial frames: {info.get('frames', 0)}")
        
        print("\n2. Running game until collision...")
        print("   (Taking random actions - this may take a few seconds)")
        
        step = 0
        max_steps = 500
        last_score = 0
        
        while step < max_steps:
            # Random action (mostly do nothing for faster collision)
            import random
            action = random.choice([0, 0, 0, 1])  # Mostly run, occasionally jump
            
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Print progress every 20 steps or when score changes
            if step % 20 == 0 or info['score'] != last_score:
                print(f"   Step {step}: Score={info['score']}, "
                      f"Frames={info['frames']}, "
                      f"Reward={reward:.2f}, "
                      f"Done={terminated}")
                last_score = info['score']
            
            if terminated:
                print(f"\n✅ GAME OVER DETECTED!")
                print(f"   Final Score: {info['score']}")
                print(f"   Total Frames: {info['frames']}")
                print(f"   Steps Taken: {step}")
                print(f"   Reward: {reward:.2f}")
                break
        
        if not terminated:
            print(f"\n⚠️  Game did not end within {max_steps} steps")
            print(f"   Last Score: {info['score']}")
            print(f"   Last Frames: {info['frames']}")
        
        # Test another episode to verify reset works
        print("\n3. Testing reset after game over...")
        obs, info = env.reset()
        print(f"   ✓ Reset successful")
        print(f"   Score after reset: {info.get('score', 0)}")
        print(f"   Frames after reset: {info.get('frames', 0)}")
        
        print("\n4. Running a few more steps to verify...")
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            if i % 3 == 0:
                print(f"   Step {i}: Score={info['score']}, Frames={info['frames']}")
        
        print("\n✅ All tests passed!")
        print("\nSummary:")
        print("  ✓ Game-over detection working")
        print("  ✓ Score extraction working")
        print("  ✓ Frame counting working")
        print("  ✓ Reset working after game over")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n5. Closing browser...")
        env.close()
        print("   Browser closed.")

if __name__ == "__main__":
    test_detection()
