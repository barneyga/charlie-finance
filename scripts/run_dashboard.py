"""Launch the Streamlit macro dashboard."""
import subprocess
import sys
from pathlib import Path

dashboard_path = Path(__file__).resolve().parent.parent / "src" / "charlie" / "viz" / "dashboard.py"

subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
