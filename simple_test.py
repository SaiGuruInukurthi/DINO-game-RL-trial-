"""Simple test - manually start game and wait before capturing"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PIL import Image
import io
import time

def simple_test():
    print("🎮 Simple Manual Test")
    print("=" * 60)
    
    # Setup browser
    chrome_options = Options()
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("\n1. Opening chromedino.com...")
        driver.get("https://chromedino.com/")
        
        print("\n2. Waiting 5 seconds for page to load...")
        time.sleep(5)
        
        print("\n3. Taking screenshot BEFORE starting game...")
        png = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        img.save('test_before_game.png')
        print(f"   Saved test_before_game.png - Size: {img.size}")
        
        print("\n4. YOU: Manually press SPACE in the browser to start the game!")
        print("   (Game should be visible in the browser window)")
        input("   Press Enter here AFTER you've started the game...")
        
        print("\n5. Taking screenshot AFTER you started game...")
        time.sleep(1)
        png = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        img.save('test_after_manual_start.png')
        print(f"   Saved test_after_manual_start.png")
        
        print("\n6. Now programmatically sending SPACE key...")
        body = driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.SPACE)
        time.sleep(0.5)
        
        png = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        img.save('test_after_space_key.png')
        print(f"   Saved test_after_space_key.png")
        
        print("\n7. Waiting 2 seconds for game to run...")
        time.sleep(2)
        
        png = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(png))
        img.save('test_after_2_seconds.png')
        print(f"   Saved test_after_2_seconds.png")
        
        print("\n8. Trying to capture just the canvas...")
        try:
            canvas = driver.find_element(By.TAG_NAME, 'canvas')
            png = canvas.screenshot_as_png
            img = Image.open(io.BytesIO(png))
            img.save('test_canvas_only.png')
            print(f"   Saved test_canvas_only.png - Size: {img.size}")
            print(f"   Canvas dimensions: {canvas.size}")
            print(f"   Canvas location: {canvas.location}")
        except Exception as e:
            print(f"   Error capturing canvas: {e}")
        
        print("\n✅ Check all test_*.png files to see what's captured!")
        input("\nPress Enter to close browser...")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    simple_test()
