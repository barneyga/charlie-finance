"""Generate weekly macro report. Run standalone or via daily_update.py."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.config import get_settings
from charlie.storage.db import Database
from charlie.analysis.report import generate_weekly_report


def main():
    parser = argparse.ArgumentParser(description="Generate Charlie Finance weekly report")
    parser.add_argument("--save", action="store_true", help="Save to reports/ directory")
    parser.add_argument("--output", "-o", type=str, help="Custom output file path")
    args = parser.parse_args()

    settings = get_settings()

    with Database(settings.db_path) as db:
        db.init_schema()
        report = generate_weekly_report(db, fred_api_key=settings.fred_api_key or "")

    if args.output:
        Path(args.output).write_text(report["markdown"], encoding="utf-8")
        print(f"Report saved to {args.output}")
    elif args.save:
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        filepath = reports_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
        filepath.write_text(report["markdown"], encoding="utf-8")
        print(f"Report saved to {filepath}")
    else:
        print(report["markdown"])


if __name__ == "__main__":
    main()
