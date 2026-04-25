from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
TARGET_BRANCH = "main"
GENERATED_PATHS = [
    ROOT / ".nojekyll",
    ROOT / "assets",
    ROOT / "gpx",
    ROOT / "previews",
    ROOT / "routes",
    ROOT / "data" / "routes",
    ROOT / "data" / "routes.json",
]


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_dist_to_pages_root() -> None:
    if not DIST.exists():
        raise SystemExit("dist/ does not exist. Run `just build` first.")

    for path in GENERATED_PATHS:
        remove_path(path)

    for child in DIST.iterdir():
        destination = ROOT / child.name
        if child.name == "data":
            (ROOT / "data").mkdir(exist_ok=True)
            for data_child in child.iterdir():
                copy_path(data_child, ROOT / "data" / data_child.name)
            continue
        copy_path(child, destination)

    (ROOT / ".nojekyll").write_text("", encoding="utf-8")


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

    copy_dist_to_pages_root()
    run(
        "git",
        "add",
        "-A",
        "index.html",
        ".nojekyll",
        "assets",
        "gpx",
        "previews",
        "routes",
        "data/routes.json",
        "data/routes",
    )
    diff = run("git", "diff", "--cached", "--quiet", check=False)
    if diff.returncode != 0:
        run("git", "commit", "-m", "Publish generated site")

    print(f"Pushing {branch} to origin/{TARGET_BRANCH} to trigger GitHub Pages.")
    result = run("git", "push", "origin", f"HEAD:{TARGET_BRANCH}", check=False)
    sys.stdout.write(result.stdout)
    if result.returncode != 0:
        return result.returncode

    print("Push complete. GitHub Pages will publish the generated root files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
