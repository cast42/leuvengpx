from pathlib import Path

from src.route_model import build_route, generate_site, slugify


def test_slugify() -> None:
    assert slugify("DR SW Eizer III") == "dr-sw-eizer-iii"


def test_build_route_from_gpx() -> None:
    route = build_route(Path("data/gpx/DR N Haacht.gpx"))

    assert route.summary.slug == "dr-n-haacht"
    assert route.summary.distance > 20
    assert route.summary.elevation_gain > 0
    assert route.summary.climb_count >= 1
    assert any(9.5 <= climb.end_distance <= 11 for climb in route.climbs)
    assert route.points


def test_max_grade_uses_distance_window() -> None:
    route = build_route(Path("data/gpx/DR SE Pallox tower.gpx"))

    assert route.climbs
    assert max(climb.max_grade for climb in route.climbs) < 15


def test_generate_site(tmp_path: Path) -> None:
    routes = generate_site(public_dir=tmp_path)

    assert routes
    assert (tmp_path / "data/routes.json").exists()
    assert (tmp_path / "data/routes/dr-n-haacht.json").exists()
    assert (tmp_path / "routes/dr-n-haacht/index.html").exists()
    assert (tmp_path / "gpx/DR N Haacht.gpx").exists()
