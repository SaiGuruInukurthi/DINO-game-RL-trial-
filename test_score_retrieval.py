"""
Test script to verify score retrieval accuracy from Chrome Dino game.
This will let the dino run straight until it hits an obstacle, then display the score.
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from browser_dino_env import BrowserDinoEnv

def test_score_retrieval():
    print("🧪 Testing Score Retrieval Accuracy")
    print("=" * 80)
    print("\n📋 Test Procedure:")
    print("1. Browser will open with Chrome Dino game")
    print("2. Dino will run straight (no jumping)")
    print("3. After collision, check the score on screen vs retrieved score")
    print("4. Test will run 3 times for verification")
    print("\n" + "=" * 80)
    
    env = BrowserDinoEnv()
    
    for test_num in range(1, 4):
        print(f"\n🔬 Test Run #{test_num}")
        print("-" * 80)
        
        # Reset environment
        state, info = env.reset()
        print("✓ Environment reset, game starting...")
        time.sleep(2)  # Wait for game to start properly
        
        # Run straight until collision (action 0 = run/do nothing)
        step_count = 0
        done = False
        
        print("🦖 Dino running straight (no actions)...")
        
        while not done and step_count < 1000:
            # Action 0 = do nothing (run straight)
            next_state, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            step_count += 1
            
            # Print periodic updates
            if step_count % 10 == 0:
                current_score = env._get_score()
                print(f"  Step {step_count}: Score = {current_score}", end='\r')
        
        print()  # New line after progress
        
        # Get final score
        final_score = env._get_score()
        
        print(f"\n📊 Results for Test #{test_num}:")
        print(f"  Steps taken: {step_count}")
        print(f"  Retrieved score: {final_score}")
        print(f"  Score from info dict: {info.get('score', 'N/A')}")
        print(f"  Total reward: {reward}")
        print(f"  Game over: {done}")
        
        print(f"\n👁️  Please check the browser window:")
        print(f"  Does the score shown match {final_score}?")
        
        # Wait for user to verify
        input("  Press Enter to continue to next test (or Ctrl+C to exit)...")
    
    # Close environment
    env.close()
    print("\n✅ Test complete!")
    print("\n📝 Analysis:")
    print("  If retrieved scores DON'T match what you see:")
    print("    - JavaScript API might be returning wrong value")
    print("    - Game speed/distance calculation issue")
    print("    - Need to check Runner.instance_.distanceRan formula")
    print("\n  If retrieved scores DO match:")
    print("    - Score retrieval is working correctly")
    print("    - High scores in training might be from longer survival")
    print("    - Check if game speed is different during training")

if __name__ == "__main__":
    try:
        test_score_retrieval()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
