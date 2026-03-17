"""Initialize the SQLite database with the schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from charlie.config import get_settings
from charlie.storage.db import Database


def main():
    settings = get_settings()
    print(f"Initializing database at {settings.db_path}")
    with Database(settings.db_path) as db:
        db.init_schema()
    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
