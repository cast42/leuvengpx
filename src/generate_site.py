from __future__ import annotations

import logging

from src.route_model import generate_site


def main() -> int:
    routes = generate_site()
    logging.info("Generated %s routes", len(routes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
