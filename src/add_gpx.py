from __future__ import annotations

import shutil
import sys
from pathlib import Path

from src.route_model import DIRECTIONS


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python -m src.add_gpx /path/to/DR N Example.gpx")
        return 2

    source = Path(sys.argv[1]).expanduser()
    if not source.exists() or source.suffix.lower() != ".gpx":
        print(f"{source} is not a GPX file")
        return 2

    parts = source.stem.split(maxsplit=2)
    if len(parts) < 3 or parts[0] != "DR" or parts[1] not in DIRECTIONS:
        print("Expected filename format: DR <direction> <name>.gpx")
        print(f"Valid directions: {', '.join(DIRECTIONS)}")
        return 2

    target = Path("data/gpx") / source.name
    if target.exists():
        print(f"{target} already exists")
        return 2
    shutil.copy2(source, target)
    print(f"Added {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
