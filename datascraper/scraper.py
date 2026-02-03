import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- CONFIGURATION ---
BASE_URL = "https://ibbi.gov.in/orders/nclt"
START_PAGE = 689
END_PAGE = 1478   # Keep this reasonable (e.g., 50 or 100)

DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "nclt_judgments")
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

TARGET_KEYWORDS = ["Final Order", "Resolution Plan", "Liquidation", "Section 12A"]

# --- CHROME SETUP ---
chrome_options = Options()
prefs = {
    "download.default_directory": DOWNLOAD_FOLDER,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "plugins.always_open_pdf_externally": True,
    "profile.default_content_settings.popups": 0
}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--disable-popup-blocking")
chrome_options.add_argument("--start-maximized")

print("🚀 Launching Chrome Bot (RAM Safe Edition)...")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def scrape_page(page_number):
    print(f"\n--- 📄 Processing Page {page_number} ---")
    driver.get(f"{BASE_URL}?page={page_number}")
    
    # Store the ID of the main tab so we can always come back
    main_window = driver.current_window_handle

    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
    except:
        print("⚠️ Table didn't load. Skipping page.")
        return

    rows = driver.find_elements(By.XPATH, "//table//tr")[1:]
    
    for i in range(len(rows)):
        try:
            # Refresh rows to prevent stale elements
            current_rows = driver.find_elements(By.XPATH, "//table//tr")[1:]
            if i >= len(current_rows): break
            
            row = current_rows[i]
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 4: continue

            remarks_text = cols[3].text
            
            if any(k.lower() in remarks_text.lower() for k in TARGET_KEYWORDS):
                
                try:
                    pdf_link = cols[2].find_element(By.TAG_NAME, "a")
                    
                    # 1. FORCE CLICK
                    driver.execute_script("arguments[0].click();", pdf_link)
                    print(f"⬇️  Triggered: {remarks_text[:30]}...")
                    
                    # 2. TAB CLEANUP (The RAM Saver)
                    time.sleep(2) # Wait for tab to open and download to start
                    
                    # Get all open tabs
                    all_windows = driver.window_handles
                    
                    # If a new tab appeared (more than 1 tab open)
                    if len(all_windows) > 1:
                        for handle in all_windows:
                            if handle != main_window:
                                driver.switch_to.window(handle)
                                driver.close() # Close the child tab
                        
                        # Switch back to main list
                        driver.switch_to.window(main_window)
                        # print("   (Closed popup tab)")

                except Exception as click_err:
                    print(f"❌ Failed row {i}: {click_err}")
                    # Ensure we are back on main window if something crashed
                    if driver.current_window_handle != main_window:
                        driver.switch_to.window(main_window)

        except Exception as e:
            continue

def main():
    for page in range(START_PAGE, END_PAGE + 1):
        scrape_page(page)
    
    print(f"\n🎉 Done! All files in: {DOWNLOAD_FOLDER}")
    time.sleep(5)
    driver.quit()

if __name__ == "__main__":
    main()