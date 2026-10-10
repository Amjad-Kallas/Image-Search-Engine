"""Capture UI screenshots for the report. Needs the API (:8000) and Streamlit (:8501) running.

Run:  uv run --with playwright python scripts/take_screenshots.py
Uses the system Edge (no browser download).
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
URL = "http://localhost:8501"


def pick_dataset(page, name):
    page.get_by_label("Dataset").click()
    page.get_by_role("option", name=name).click()
    page.wait_for_timeout(500)


def wait_results(page, n_images=10):
    page.wait_for_function(
        f"document.querySelectorAll('[data-testid=stImage] img').length >= {n_images}", timeout=90000
    )
    page.wait_for_timeout(1500)  # let images finish loading


with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge")
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(URL)
    page.get_by_text("Search by text").wait_for()

    # 1) text -> image on COCO
    page.get_by_placeholder("a dog catching a frisbee").fill("a red double decker bus")
    page.keyboard.press("Enter")
    wait_results(page)
    page.screenshot(path=OUT / "ui-text-search.png", full_page=True)

    # 2) image -> image on Pets
    page.reload()
    page.get_by_text("Search by image").click()
    pick_dataset(page, "pets")
    page.set_input_files("input[type=file]", str(ROOT / "data" / "pets" / "jpg" / "Birman_10.jpg"))
    wait_results(page, 11)  # 10 results + the query preview
    page.screenshot(path=OUT / "ui-image-search.png", full_page=True)
    browser.close()
print("saved")
