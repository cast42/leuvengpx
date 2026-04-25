from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageDraw, ImageFont

START_LOCATION = (50.876777, 4.715101)
DIRECTIONS = ("NW", "N", "NE", "W", "C", "E", "SW", "S", "SE")
SITE_URL = "https://cast42.github.io/leuvengpx"


@dataclass(frozen=True)
class TrackPoint:
    lat: float
    lon: float
    elevation: float
    distance: float


@dataclass(frozen=True)
class Climb:
    id: int
    start_index: int
    end_index: int
    start_distance: float
    end_distance: float
    length: float
    ascent: float
    grade: float
    score: float
    category: int
    lat: float
    lon: float
    max_grade: float


@dataclass(frozen=True)
class RouteSummary:
    slug: str
    name: str
    file_name: str
    direction: str
    distance: float
    elevation_gain: float
    climb_count: int
    start: tuple[float, float]
    end: tuple[float, float]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    detail_url: str
    gpx_url: str
    icon: str


@dataclass(frozen=True)
class RouteDetail:
    summary: RouteSummary
    points: list[TrackPoint]
    climbs: list[Climb]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "route"


def route_name_from_file(path: Path) -> tuple[str, str]:
    parts = path.stem.split(maxsplit=2)
    if len(parts) < 3:
        return path.stem, "C"
    prefix, direction, name = parts
    if prefix != "DR" or direction not in DIRECTIONS:
        return path.stem, "C"
    return name, direction


def parse_gpx(path: Path) -> tuple[list[float], list[float], list[float]]:
    tree = ElementTree.parse(path)
    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", maxsplit=1)[0] + "}"

    latitudes: list[float] = []
    longitudes: list[float] = []
    elevations: list[float] = []
    for point in root.iter(f"{namespace}trkpt"):
        elevation = point.find(f"{namespace}ele")
        if elevation is None or elevation.text is None:
            continue
        latitudes.append(float(point.attrib["lat"]))
        longitudes.append(float(point.attrib["lon"]))
        elevations.append(float(elevation.text))

    if len(latitudes) < 2:
        msg = f"{path} does not contain enough GPX track points"
        raise ValueError(msg)
    return latitudes, longitudes, elevations


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cumulative_distances(latitudes: list[float], longitudes: list[float]) -> np.ndarray:
    distances = [0.0]
    for index in range(1, len(latitudes)):
        distances.append(
            distance_km(
                latitudes[index - 1],
                longitudes[index - 1],
                latitudes[index],
                longitudes[index],
            )
        )
    return np.cumsum(np.array(distances))


def moving_median(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        result[index] = float(np.median(values[start:end]))
    return result


def clean_elevation(elevations: list[float]) -> np.ndarray:
    values = np.array(elevations, dtype=float)
    window = min(21, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 3:
        return values
    return moving_median(values, window)


def climb_category(score: float) -> int:
    if score < 1_500:
        return 5
    if score < 8_000:
        return 4
    if score < 16_000:
        return 3
    if score < 32_000:
        return 2
    if score < 64_000:
        return 1
    return 0


def find_peaks_and_valleys(elevations: np.ndarray) -> tuple[list[int], list[int]]:
    peaks: list[int] = []
    valleys: list[int] = []
    if len(elevations) < 2:
        return peaks, valleys

    initial_direction_index = 1
    while (
        initial_direction_index < len(elevations)
        and elevations[initial_direction_index] == elevations[0]
    ):
        initial_direction_index += 1

    if initial_direction_index >= len(elevations):
        return peaks, valleys

    direction = np.sign(elevations[initial_direction_index] - elevations[0])
    if direction == 1:
        valleys.append(0)
    elif direction == -1:
        peaks.append(0)

    for index in range(initial_direction_index, len(elevations) - 1):
        new_direction = np.sign(elevations[index + 1] - elevations[index])
        if new_direction == 0 or new_direction == direction:
            continue
        if direction == 1 and new_direction == -1:
            peaks.append(index)
        elif direction == -1 and new_direction == 1:
            valleys.append(index)
        direction = new_direction

    if direction == 1:
        peaks.append(len(elevations) - 1)
    elif direction == -1:
        valleys.append(len(elevations) - 1)

    return peaks, valleys


def max_window_grade(
    distances: np.ndarray,
    elevations: np.ndarray,
    start_index: int,
    end_index: int,
    window_meters: float = 250,
) -> float:
    max_grade = 0.0
    for index in range(start_index, end_index):
        target_distance = distances[index] + window_meters / 1000
        window_end = int(np.searchsorted(distances, target_distance, side="left"))
        window_end = min(window_end, end_index)
        if window_end <= index:
            continue
        distance_meters = (distances[window_end] - distances[index]) * 1000
        if distance_meters <= 0:
            continue
        grade = (elevations[window_end] - elevations[index]) / distance_meters * 100
        max_grade = max(max_grade, float(grade))
    return max_grade


def detect_climbs(
    latitudes: list[float],
    longitudes: list[float],
    distances: np.ndarray,
    elevations: np.ndarray,
) -> list[Climb]:
    if len(elevations) < 10:
        return []

    peaks, valleys = find_peaks_and_valleys(elevations)
    climbs: list[Climb] = []
    valley_cursor = 0
    climb_id = 1
    for peak_index in peaks:
        valley_index = -1
        while valley_cursor < len(valleys) and valleys[valley_cursor] < peak_index:
            valley_index = valleys[valley_cursor]
            valley_cursor += 1

        if valley_index == -1:
            continue

        length = float(distances[peak_index] - distances[valley_index])
        ascent = float(elevations[peak_index] - elevations[valley_index])
        if length <= 0 or ascent <= 0:
            continue

        grade = ascent / (length * 1000) * 100
        score = length * 1000 * grade
        if score < 1_500 or length < 0.25:
            continue

        max_grade = max_window_grade(distances, elevations, valley_index, peak_index)

        climbs.append(
            Climb(
                id=climb_id,
                start_index=valley_index,
                end_index=peak_index,
                start_distance=float(distances[valley_index]),
                end_distance=float(distances[peak_index]),
                length=length,
                ascent=ascent,
                grade=grade,
                score=score,
                category=climb_category(score),
                lat=latitudes[peak_index],
                lon=longitudes[peak_index],
                max_grade=max(max_grade, grade),
            )
        )
        climb_id += 1
    return climbs


def route_icon(direction: str) -> str:
    return {
        "N": "arrow-up",
        "NE": "arrow-up-right",
        "E": "arrow-right",
        "SE": "arrow-down-right",
        "S": "arrow-down",
        "SW": "arrow-down-left",
        "W": "arrow-left",
        "NW": "arrow-up-left",
    }.get(direction, "circle-dot")


def build_route(path: Path) -> RouteDetail:
    latitudes, longitudes, raw_elevations = parse_gpx(path)
    elevations = clean_elevation(raw_elevations)
    distances = cumulative_distances(latitudes, longitudes)
    name, direction = route_name_from_file(path)
    slug = slugify(path.stem)
    points = [
        TrackPoint(
            lat=round(latitudes[index], 6),
            lon=round(longitudes[index], 6),
            elevation=round(float(elevations[index]), 1),
            distance=round(float(distances[index]), 3),
        )
        for index in range(len(latitudes))
    ]
    climbs = detect_climbs(latitudes, longitudes, distances, elevations)
    elevation_gain = float(np.maximum(np.diff(elevations), 0).sum())
    summary = RouteSummary(
        slug=slug,
        name=name,
        file_name=path.name,
        direction=direction,
        distance=round(float(distances[-1]), 1),
        elevation_gain=round(elevation_gain),
        climb_count=len(climbs),
        start=(round(latitudes[0], 6), round(longitudes[0], 6)),
        end=(round(latitudes[-1], 6), round(longitudes[-1], 6)),
        bounds=(
            (round(min(latitudes), 6), round(min(longitudes), 6)),
            (round(max(latitudes), 6), round(max(longitudes), 6)),
        ),
        detail_url=f"routes/{slug}/",
        gpx_url=f"gpx/{path.name}",
        icon=route_icon(direction),
    )
    return RouteDetail(summary=summary, points=points, climbs=climbs)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def preview_html(route: RouteSummary) -> str:
    title = f"{route.name} - {route.distance:.1f} km from Leuven"
    description = (
        f"{route.direction} route, {route.elevation_gain:.0f} m elevation gain, "
        f"{route.climb_count} detected climbs. Download the GPX and inspect the map."
    )
    url = f"{SITE_URL}/{route.detail_url}"
    image_url = f"{SITE_URL}/previews/{route.slug}.png"
    redirect = f"../../?route={route.slug}"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <meta name="description" content="{description}" />
    <meta property="og:type" content="article" />
    <meta property="og:title" content="{title}" />
    <meta property="og:description" content="{description}" />
    <meta property="og:url" content="{url}" />
    <meta property="og:image" content="{image_url}" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:image" content="{image_url}" />
    <meta http-equiv="refresh" content="0; url={redirect}" />
    <script>window.location.replace("{redirect}")</script>
  </head>
  <body>
    <p><a href="{redirect}">Open {route.name}</a></p>
  </body>
</html>
"""


def generate_preview_image(route: RouteSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1200, 630), "#fafaf9")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=74)
    stat_font = ImageFont.load_default(size=44)
    label_font = ImageFont.load_default(size=30)

    draw.rectangle((0, 0, 1200, 630), fill="#fafaf9")
    draw.rectangle((0, 0, 1200, 18), fill="#0f766e")
    draw.text((72, 76), "Leuven GPX", fill="#0f766e", font=label_font)
    draw.text((72, 138), route.name, fill="#1c1917", font=title_font)
    draw.text((72, 236), f"{route.direction} route", fill="#57534e", font=stat_font)

    stats = [
        ("Distance", f"{route.distance:.1f} km"),
        ("Elevation", f"{route.elevation_gain:.0f} m"),
        ("Climbs", str(route.climb_count)),
    ]
    for index, (label, value) in enumerate(stats):
        left = 72 + index * 350
        draw.rounded_rectangle(
            (left, 360, left + 290, 510),
            radius=18,
            fill="#ffffff",
            outline="#e7e5e4",
            width=2,
        )
        draw.text((left + 28, 392), label, fill="#78716c", font=label_font)
        draw.text((left + 28, 438), value, fill="#1c1917", font=stat_font)

    draw.text(
        (72, 562),
        "Map, elevation profile, climbs and GPX download",
        fill="#57534e",
        font=label_font,
    )
    image.save(path)


def generate_site(
    gpx_dir: Path = Path("data/gpx"),
    public_dir: Path = Path("public"),
) -> list[RouteDetail]:
    routes = sorted(
        (build_route(path) for path in gpx_dir.glob("*.gpx")),
        key=lambda route: (route.summary.direction, route.summary.name),
    )
    data_dir = public_dir / "data"
    route_data_dir = data_dir / "routes"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if (public_dir / "routes").exists():
        shutil.rmtree(public_dir / "routes")
    if (public_dir / "gpx").exists():
        shutil.rmtree(public_dir / "gpx")
    if (public_dir / "previews").exists():
        shutil.rmtree(public_dir / "previews")
    (public_dir / "gpx").mkdir(parents=True, exist_ok=True)

    summaries = [asdict(route.summary) for route in routes]
    write_json(data_dir / "routes.json", summaries)
    for route in routes:
        shutil.copy2(
            gpx_dir / route.summary.file_name,
            public_dir / route.summary.gpx_url,
        )
        write_json(
            route_data_dir / f"{route.summary.slug}.json",
            {
                "summary": asdict(route.summary),
                "points": [asdict(point) for point in route.points],
                "climbs": [asdict(climb) for climb in route.climbs],
            },
        )
        preview_path = public_dir / "routes" / route.summary.slug / "index.html"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(preview_html(route.summary), encoding="utf-8")
        generate_preview_image(
            route.summary,
            public_dir / "previews" / f"{route.summary.slug}.png",
        )
    return routes
