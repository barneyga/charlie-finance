"""Take a full-page screenshot of the Charlie Finance dashboard.

Usage:
    python scripts/screenshot.py                       # save to screenshots/YYYY-MM-DD.png
    python scripts/screenshot.py -o my_screenshot.png  # custom output path
    python scripts/screenshot.py --width 1920          # custom viewport width

Requires: pip install playwright && python -m playwright install chromium
The dashboard must already be running (streamlit run src/charlie/viz/dashboard.py).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CSS injected to break Streamlit out of its fixed-height scroll container
_UNLOCK_CSS = """
    .stApp, .stApp > div, .stMain,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stMainBlockContainer"],
    section.main, section.main > div {
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
        overflow: visible !important;
        position: static !important;
    }
    header[data-testid="stHeader"],
    [data-testid="stSidebar"],
    [data-testid="stToolbar"],
    .stDeployButton { display: none !important; }
"""


def main():
    parser = argparse.ArgumentParser(description="Screenshot the Charlie Finance dashboard")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port (default: 8501)")
    parser.add_argument("--width", type=int, default=1400, help="Viewport width (default: 1400)")
    parser.add_argument("--wait", type=int, default=10, help="Seconds to wait for page load (default: 10)")
    parser.add_argument("--start", action="store_true", help="Auto-start Streamlit (will stop it after)")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")
        sys.exit(1)

    if args.output:
        output = Path(args.output)
    else:
        screenshots_dir = PROJECT_ROOT / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        output = screenshots_dir / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.png"

    url = f"http://localhost:{args.port}/?screenshot=true"
    server_proc = None

    if args.start:
        dashboard_path = PROJECT_ROOT / "src" / "charlie" / "viz" / "dashboard.py"
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
             "--server.port", str(args.port), "--server.headless", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"Starting Streamlit on port {args.port}...")
        time.sleep(8)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": args.width, "height": 900})
            print(f"Loading {url} ...")
            page.goto(url, wait_until="networkidle", timeout=60000)

            print(f"Waiting {args.wait}s for charts to render...")
            time.sleep(args.wait)

            # Inject CSS to unlock scroll containers
            page.add_style_tag(content=_UNLOCK_CSS)
            time.sleep(1)

            # Scroll down in steps to force all lazy content to render
            for _ in range(50):
                page.evaluate("window.scrollBy(0, 800)")
                time.sleep(0.15)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1)

            # Re-inject CSS after scroll (Streamlit may re-render)
            page.add_style_tag(content=_UNLOCK_CSS)
            time.sleep(1)

            # Measure full content height
            total_height = page.evaluate("""
                Math.max(
                    document.body.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.scrollHeight,
                    document.documentElement.offsetHeight
                )
            """)
            print(f"Page height: {total_height}px")

            # Resize viewport to full page height so screenshot captures everything
            page.set_viewport_size({"width": args.width, "height": total_height})
            time.sleep(2)

            # Re-inject one more time after resize
            page.add_style_tag(content=_UNLOCK_CSS)
            time.sleep(1)

            # Screenshot with full_page=True
            page.screenshot(path=str(output), full_page=True)

            file_size = output.stat().st_size
            print(f"Screenshot saved to {output}")
            print(f"Size: {file_size / 1024:.0f} KB ({file_size / 1024 / 1024:.1f} MB)")

            browser.close()

    finally:
        if server_proc:
            server_proc.terminate()
            server_proc.wait()
            print("Streamlit stopped.")


if __name__ == "__main__":
    main()
