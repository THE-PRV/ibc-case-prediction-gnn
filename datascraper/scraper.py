"""
IBBI NCLT Judgment Scraper

Downloads PDF judgments from the IBBI (Insolvency and Bankruptcy Board of India)
orders portal, filtering for IBC-relevant document types.

Target keywords (configured in ScraperConfig):
    - "Final Order"
    - "Resolution Plan"
    - "Liquidation"
    - "Section 12A"

Usage:
    python -m datascraper.scraper                   # use config defaults
    python -m datascraper.scraper --start 689 --end 700
"""

import argparse
import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def _build_chrome_driver(download_folder: str) -> webdriver.Chrome:
    """
    Create and return a configured Chrome WebDriver instance.

    Chrome is set to auto-download PDFs (bypassing the built-in viewer) and
    to suppress popup dialogs so downloads start without user interaction.

    Args:
        download_folder: Absolute path where downloaded PDFs will be saved.

    Returns:
        A ready-to-use Chrome WebDriver.
    """
    os.makedirs(download_folder, exist_ok=True)

    prefs = {
        "download.default_directory": download_folder,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        # Open PDFs externally (triggers download rather than in-browser view)
        "plugins.always_open_pdf_externally": True,
        "profile.default_content_settings.popups": 0,
    }

    options = Options()
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--start-maximized")

    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )


def scrape_page(driver: webdriver.Chrome, page_number: int,
                base_url: str, target_keywords: list) -> None:
    """
    Scrape a single IBBI orders listing page and download matching PDFs.

    Each table row is inspected for keyword matches in the remarks column.
    Matching rows have their PDF link clicked, which triggers a download.
    Any popup tab opened by the click is immediately closed to keep RAM usage
    manageable across hundreds of pages.

    Args:
        driver:          Active Chrome WebDriver session.
        page_number:     Page index to fetch (appended as ?page=N).
        base_url:        Base URL of the IBBI orders portal.
        target_keywords: List of strings; a row matches if any keyword appears
                         (case-insensitive) in its remarks column.
    """
    print(f"\n--- Processing Page {page_number} ---")
    driver.get(f"{base_url}?page={page_number}")

    main_window = driver.current_window_handle

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )
    except Exception:
        print(f"  Warning: Table did not load on page {page_number}, skipping.")
        return

    # Skip header row (index 0)
    rows = driver.find_elements(By.XPATH, "//table//tr")[1:]

    for i in range(len(rows)):
        try:
            # Re-fetch rows each iteration to avoid stale element references
            current_rows = driver.find_elements(By.XPATH, "//table//tr")[1:]
            if i >= len(current_rows):
                break

            row = current_rows[i]
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 4:
                continue

            remarks_text = cols[3].text
            if not any(k.lower() in remarks_text.lower() for k in target_keywords):
                continue

            try:
                pdf_link = cols[2].find_element(By.TAG_NAME, "a")
                # Use JS click to bypass potential overlay elements
                driver.execute_script("arguments[0].click();", pdf_link)
                print(f"  Downloaded: {remarks_text[:60]}...")

                # Wait briefly for the browser to open a new tab for the PDF
                time.sleep(2)

                # Close any popup tabs immediately to prevent RAM accumulation
                all_windows = driver.window_handles
                if len(all_windows) > 1:
                    for handle in all_windows:
                        if handle != main_window:
                            driver.switch_to.window(handle)
                            driver.close()
                    driver.switch_to.window(main_window)

            except Exception as click_err:
                print(f"  Failed to click row {i}: {click_err}")
                # Recover to main window if the click left us on a different tab
                if driver.current_window_handle != main_window:
                    driver.switch_to.window(main_window)

        except Exception:
            # Swallow stale-element and similar transient errors; continue to next row
            continue


def main():
    """Parse arguments, initialise the browser, and scrape the requested page range."""
    # Import config here to avoid a circular import if this module is used standalone
    try:
        from src.utils.config import scraper_config
        default_base_url = scraper_config.base_url
        default_start = scraper_config.start_page
        default_end = scraper_config.end_page
        default_download = str(scraper_config.download_folder)
        default_keywords = scraper_config.target_keywords
    except ImportError:
        # Fallback defaults when run directly without the src package on the path
        default_base_url = "https://ibbi.gov.in/orders/nclt"
        default_start = 689
        default_end = 1478
        default_download = os.path.join(os.getcwd(), "data", "nclt_judgments")
        default_keywords = ["Final Order", "Resolution Plan", "Liquidation", "Section 12A"]

    parser = argparse.ArgumentParser(description="IBBI NCLT Judgment Scraper")
    parser.add_argument("--start", type=int, default=default_start, help="First page to scrape")
    parser.add_argument("--end", type=int, default=default_end, help="Last page to scrape (inclusive)")
    parser.add_argument("--url", default=default_base_url, help="Base URL of the IBBI orders portal")
    parser.add_argument("--output", default=default_download, help="Folder to save downloaded PDFs")
    args = parser.parse_args()

    print(f"Starting scraper: pages {args.start}–{args.end}")
    print(f"Saving PDFs to: {args.output}")

    driver = _build_chrome_driver(args.output)

    try:
        for page in range(args.start, args.end + 1):
            scrape_page(driver, page, args.url, default_keywords)
    finally:
        # Always quit the driver, even if an exception occurs mid-scrape
        time.sleep(3)
        driver.quit()
        print(f"\nDone. Files saved to: {args.output}")


if __name__ == "__main__":
    main()
