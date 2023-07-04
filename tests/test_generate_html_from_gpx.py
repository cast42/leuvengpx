from pathlib import Path

# from src import generate_html_from_gpx


def test_overview_map():
    # assert overview_map.main() == 0
    assert Path("data/html/map.html").exists()
    assert Path("index.html").exists()
