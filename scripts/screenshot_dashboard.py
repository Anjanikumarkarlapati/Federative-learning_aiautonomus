"""Screenshot every dashboard page into docs/run_guide/ (needs a running dashboard).

    streamlit run dashboard/app.py          # in another terminal
    python scripts/screenshot_dashboard.py [--url http://localhost:8501] [--skip-launch]

Set CHROMIUM_PATH to a Chromium binary if Playwright has no downloaded browser.
"""
from __future__ import annotations

import argparse
import os

from playwright.sync_api import sync_playwright

PAGES = [
    ("Overview", "step10_dashboard_overview.png"),
    ("Fleet (every vehicle)", "step11_dashboard_fleet.png"),
    ("Benchmark", "step12_dashboard_benchmark.png"),
    ("Ablation", "step13_dashboard_ablation.png"),
    ("Launch a run", "step14_dashboard_launch.png"),
    ("Run explorer", "step16_dashboard_explorer.png"),
    ("Validation", "step17_dashboard_validation.png"),
]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "run_guide")


def wait_idle(page) -> None:
    page.wait_for_timeout(1500)
    page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=120_000)
    page.wait_for_timeout(2500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--max-height", type=int, default=2400)
    ap.add_argument("--skip-launch", action="store_true", help="don't start a run from the Launch page")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with sync_playwright() as p:
        exe = os.environ.get("CHROMIUM_PATH")
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(a.url)
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=60_000)
        wait_idle(page)
        for label, fname in PAGES:
            page.locator('[data-testid="stSidebar"] label').filter(has_text=label).first.click()
            wait_idle(page)
            # Streamlit scrolls inside its main container, so size the viewport to the content.
            height = page.evaluate(
                "() => { const m = document.querySelector('[data-testid=\"stMain\"]')"
                " || document.querySelector('section.main'); return m ? m.scrollHeight : 900; }")
            page.set_viewport_size({"width": 1440, "height": max(900, min(int(height), a.max_height))})
            wait_idle(page)
            page.screenshot(path=os.path.join(OUT, fname))
            page.set_viewport_size({"width": 1440, "height": 900})
            print("saved", fname, height)
        if not a.skip_launch:
            launch_and_capture(page, a.max_height)
        browser.close()


def launch_and_capture(page, max_height: int) -> None:
    """Start a quick run from the Launch page and screenshot it once it has finished."""
    page.locator('[data-testid="stSidebar"] label').filter(has_text="Launch a run").first.click()
    wait_idle(page)
    page.get_by_role("button", name="Start run").click()
    page.get_by_text("Finished —").wait_for(timeout=600_000)
    page.wait_for_timeout(4000)
    height = page.evaluate("() => document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
    page.set_viewport_size({"width": 1440, "height": max(900, min(int(height), max_height + 400))})
    page.wait_for_timeout(3000)
    page.screenshot(path=os.path.join(OUT, "step15_dashboard_launch_finished.png"))
    print("saved step15_dashboard_launch_finished.png", height)


if __name__ == "__main__":
    main()
