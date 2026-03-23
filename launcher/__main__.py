"""
__main__.py — Entry point: python -m launcher  or  python launcher/
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Model Server Launcher")
    parser.add_argument(
        "--no-menu",
        action="store_true",
        help="Skip menus and launch with default settings",
    )
    args = parser.parse_args()
    from launcher.app import run
    run(no_menu=args.no_menu)


if __name__ == "__main__":
    main()
