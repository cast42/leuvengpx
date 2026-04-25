from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_BRANCH = "main"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    status = run("git", "status", "--porcelain")
    if status.stdout.strip():
        print("Refusing to publish with uncommitted changes.")
        print("Commit your changes first, then run `just publish` again.")
        return 1

    branch = run("git", "branch", "--show-current").stdout.strip()
    if not branch:
        print("Refusing to publish from a detached HEAD.")
        return 1

    print(f"Pushing {branch} to origin/{TARGET_BRANCH} to trigger GitHub Pages.")
    result = run("git", "push", "origin", f"HEAD:{TARGET_BRANCH}", check=False)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        return result.returncode

    print("Push complete. The GitHub Pages workflow will build and deploy the site.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
