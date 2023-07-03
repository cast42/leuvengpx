from pathlib import Path

from src import overview_map


def test_overview_map():
    assert overview_map.main() == 0
    assert Path("data/html/map.html").exists()
    assert Path("index.html").exists()
