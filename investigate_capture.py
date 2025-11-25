"""Investigate why screenshots are mostly blank"""

from src.browser_dino_env import BrowserDinoEnv
import time
from PIL import Image
import io

def investigate_capture():
    print("🔍 Investigating Screenshot Issue")
    print("=" * 60)
    
    env = BrowserDinoEnv(headless=False, target_fps=20)
    
    try:
        print("\n1. Waiting for game to load...")
        time.sleep(3)
        
        print("\n2. Checking what elements we can find...")
        # Check canvas
        try:
            canvas = env.driver.find_element_by_tag_name('canvas')
            print(f"   ✓ Canvas found: {canvas.size}")
            print(f"   Canvas location: {canvas.location}")
        except:
            print("   ✗ No canvas found!")
        
        print("\n3. Taking full page screenshot to see everything...")
        png = env.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        img.save('debug_full_page.png')
        print(f"   Saved debug_full_page.png - Size: {img.size}")
        
        print("\n4. Getting page HTML structure...")
        html = env.driver.execute_script("return document.body.innerHTML;")
        with open('debug_page_structure.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print("   Saved debug_page_structure.html")
        
        print("\n5. Checking canvas content...")
        canvas_info = env.driver.execute_script("""
            var canvas = document.querySelector('canvas');
            if (canvas) {
                return {
                    width: canvas.width,
                    height: canvas.height,
                    style: canvas.style.cssText,
                    visible: canvas.offsetWidth > 0 && canvas.offsetHeight > 0
                };
            }
            return null;
        """)
        print(f"   Canvas info: {canvas_info}")
        
        print("\n6. Starting the game...")
        env.reset()
        time.sleep(2)
        
        print("\n7. Taking another full page screenshot after game start...")
        png = env.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        img.save('debug_after_start.png')
        print(f"   Saved debug_after_start.png - Size: {img.size}")
        
        print("\n8. Taking several steps and capturing...")
        for i in range(5):
            env.step(0)  # Do nothing
            time.sleep(0.2)
        
        png = env.driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        img.save('debug_after_steps.png')
        print(f"   Saved debug_after_steps.png")
        
        print("\n✅ Check the debug_*.png and debug_page_structure.html files")
        print("   to see what's actually being captured!")
        
    finally:
        input("\nPress Enter to close browser...")
        env.close()

if __name__ == "__main__":
    investigate_capture()
