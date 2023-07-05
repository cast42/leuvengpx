import json
import sys
from pathlib import Path
from typing import List
from urllib.request import pathname2url
from xml.dom import minidom

import altair as alt
import branca
import folium
import numpy as np
import pandas as pd
from folium import features
from jinja2 import Environment, FileSystemLoader
from scipy.signal import find_peaks


def get_gpx(filepath: str):
    data = open(filepath)
    xmldoc = minidom.parse(data)
    track = xmldoc.getElementsByTagName("trkpt")
    elevation = xmldoc.getElementsByTagName("ele")
    n_track = len(track)

    # Parsing GPX elements
    lon_list = []
    lat_list = []
    h_list = []
    for s in range(n_track):
        lon, lat = (
            track[s].attributes["lon"].value,
            track[s].attributes["lat"].value,
        )
        elev = elevation[s].firstChild.nodeValue
        lon_list.append(float(lon))
        lat_list.append(float(lat))
        h_list.append(float(elev))

    # Calculate average latitude and longitude
    ave_lat = sum(lat_list) / len(lat_list)
    ave_lon = sum(lon_list) / len(lon_list)

    return ave_lat, ave_lon, lon_list, lat_list, h_list


# From https://tomaugspurger.net/posts/modern-4-performance/
def gcd_vec(lat1, lng1, lat2, lng2):
    """
    Calculate great circle distance.
    http://www.johndcook.com/blog/python_longitude_latitude/

    Parameters
    ----------
    lat1, lng1, lat2, lng2: float or array of float

    Returns
    -------
    distance:
      distance from ``(lat1, lng1)`` to ``(lat2, lng2)`` in kilometers.
    """
    # python2 users will have to use ascii identifiers
    ϕ1 = np.deg2rad(90 - lat1)
    ϕ2 = np.deg2rad(90 - lat2)

    θ1 = np.deg2rad(lng1)
    θ2 = np.deg2rad(lng2)

    cos = np.sin(ϕ1) * np.sin(ϕ2) * np.cos(θ1 - θ2) + np.cos(ϕ1) * np.cos(ϕ2)
    arc = np.arccos(cos)
    return arc * 6373


CATEGORY_TO_COLOR = {
    5: "#68bd44",
    4: "#68bd44",
    3: "#fbaa1c",
    2: "#f15822",
    1: "#ed2125",
    0: "#800000",
}


def climb_category(climb_score):
    """Determine category of the climb based on the climb score as defined by Garmin"""
    if climb_score < 1_500:
        return 5  # Not categorised
    elif climb_score < 8_000:
        return 4
    elif climb_score < 16_000:
        return 3
    elif climb_score < 32000:
        return 2
    elif climb_score < 64000:
        return 1
    else:
        return 0  # Hors categorie


def grade_to_color(grade):
    """Determine the color of the climb based on its grade according to Garmin"""
    if grade < 3:
        return "lightgrey"
    elif grade < 6:
        return CATEGORY_TO_COLOR[3]
    elif grade < 9:
        return CATEGORY_TO_COLOR[2]
    elif grade < 12:
        return CATEGORY_TO_COLOR[1]
    else:
        return CATEGORY_TO_COLOR[0]


def find_climbs(df: pd.DataFrame) -> pd.DataFrame:
    peaks, _ = find_peaks(df["smoothed_elevation"])
    df_peaks = df.iloc[peaks, :].assign(base=0).assign(kind="peak")
    valleys, _ = find_peaks(df["smoothed_elevation"].max() - df["smoothed_elevation"])
    df_valleys = df.iloc[valleys, :].assign(base=0).assign(kind="valley")
    df_elevation = pd.concat([df_valleys, df_peaks], axis=0).sort_values(
        by="distance_from_start"
    )

    # Climbscore acoording to Garmin:
    # https://s3.eu-central-1.amazonaws.com/download.navigation-professionell.de/
    # Garmin/Manuals/Understanding+ClimbPro+on+the+Edge.pdf

    df_peaks_filtered = (
        pd.concat(
            [df_elevation, df_elevation.shift(1).bfill().add_prefix("prev_")],
            axis=1,
        )
        .query("(kind=='peak') & (prev_kind=='valley')")
        .assign(
            length=lambda df_: df_["distance_from_start"]
            - df_["prev_distance_from_start"]
        )
        .assign(total_ascent=lambda df_: df_["elev"] - df_["prev_elev"])
        .assign(grade=lambda df_: (df_["total_ascent"] / df_["length"]) * 100)
        .assign(climb_score=lambda df_: df_["length"] * df_["grade"])
        .assign(hill_category=lambda df_: df_["climb_score"].map(climb_category))
        .query("climb_score >= 1_500")
        .assign(max_elevation=df["elev"].max().round(-2) + 10)
    )
    # Garmin rules
    # df_peaks_filtered = df_peaks_meta.query(
    #   "(climb_score >= 1_500) & (length >= 0.5) & (grade >= 3_000)"
    # )
    return df_peaks_filtered


def generate_height_profile_json(df: pd.DataFrame) -> str:
    """Generate a height profile of the ride in Altair.
    Returns a string with json.
    """
    df_distance = (
        df.assign(lon_1=lambda df_: df["lon"].shift(1))
        .assign(lat_1=lambda df_: df["lat"].shift(1))
        .drop(columns=["elev"])
    )[["lat", "lon", "lat_1", "lon_1"]]
    df["distance"] = pd.Series(
        [gcd_vec(*x) for x in df_distance.itertuples(index=False)],
        index=df_distance.index,
    ).fillna(0)
    total_distance = df["distance"].sum()
    total_distance_round = np.round(total_distance)
    df["distance_from_start"] = df["distance"].cumsum()
    df["smoothed_elevation"] = df["elev"].rolling(10).mean().bfill()
    df["grade"] = (
        0.1
        * (df["elev"] - df["elev"].shift(1).bfill())
        / (df["distance_from_start"] - df["distance_from_start"].shift(1).bfill())
    )
    df["smoothed_grade"] = df["grade"].rolling(10).mean()
    df["smoothed_grade"] = df["smoothed_grade"].bfill()
    df["smoothed_grade_color"] = df["smoothed_grade"].map(grade_to_color)
    # df["grade_color"] = df["grade"].map(grade_to_color)

    elevation = (
        alt.Chart(
            df[["distance_from_start", "smoothed_elevation", "smoothed_grade_color"]]
        )
        .mark_bar()
        .encode(
            x=alt.X("distance_from_start")
            .axis(
                grid=False,
                tickCount=10,
                labelExpr="datum.label + ' km'",
                title=None,
            )
            .scale(domain=(0, total_distance_round)),
            y=alt.Y("smoothed_elevation").axis(
                domain=False,
                ticks=False,
                tickCount=5,
                labelExpr="datum.label + ' m'",
                title=None,
            ),
            color=alt.Color("smoothed_grade_color").scale(None),
        )
    )

    df_peaks_filtered = find_climbs(df)
    line_peaks = (
        alt.Chart(df_peaks_filtered[["distance_from_start", "elev", "max_elevation"]])
        .mark_rule(color="red")
        .encode(
            x=alt.X("distance_from_start:Q").scale(domain=(0, total_distance_round)),
            y="elev",
            y2="max_elevation",
        )
    )
    chart = (
        (elevation + line_peaks)
        .properties(width="container")
        .configure_view(
            strokeWidth=0,
        )
    )
    return chart.to_json()


def main(cli_args: List[str] = None) -> int:
    templates_dir = "data/templates"
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("index.html")

    gpx_file_list = []
    gpx_name_list = []
    for gpxfile in Path("data/gpx/").glob("*.gpx"):
        gpx_file_list.append(pathname2url(gpxfile.stem) + ".html")
        gpx_name_list.append(gpxfile.stem)

    print(gpx_file_list)
    print(gpx_name_list)

    with open("index.html", "w") as fh:
        fh.write(
            template.render(
                names=zip(gpx_name_list, gpx_file_list),
            )
        )

    # Create a folium map
    my_map = folium.Map(location=[50.876777, 4.715101], zoom_start=10)

    for gpxfile in Path("data/gpx/").glob("*.gpx"):
        # Open gpx file and parse its content
        ave_lat, ave_lon, lon_list, lat_list, h_list = get_gpx(gpxfile)
        # Store coordinates and elevation in a Pandas dataframe
        df = pd.DataFrame({"lon": lon_list, "lat": lat_list, "elev": h_list})

        color = "red" if gpxfile.name == "DR SW Huldenberg.gpx" else "#38b580"
        # Add a polyline to connect the track points
        folium.PolyLine(
            list(zip(lat_list, lon_list)), color=color, weight=2.5, opacity=0.8
        ).add_to(my_map)

        f = branca.element.Figure()
        route_map = folium.Map(
            location=[ave_lat, ave_lon],
            zoom_start=12,
            position="absolute",
            left="0%",
            width="100%",
            height="70%",
        )
        folium.PolyLine(
            list(zip(lat_list, lon_list)), color="red", weight=2.5, opacity=1
        ).add_to(route_map)
        f.add_child(route_map)

        chart_json = generate_height_profile_json(df)
        height_profile = features.VegaLite(
            json.loads(chart_json),
            position="absolute",
            left="0%",
            width="100%",
            height="100%",
            top="70%",
        )
        f.add_child(height_profile)
        html_file = f"data/html/{gpxfile.stem}.html"
        f.save(html_file)

        json_file_path = f"data/html/{gpxfile.stem}.json"
        with open(json_file_path, "w") as json_file:
            json_file.write(chart_json)
        route_template = env.get_template("route.html")
        with open(f"data/html/route_{gpxfile.stem}.html", "w") as fh:
            fh.write(
                route_template.render(
                    json_file_path=pathname2url(json_file_path),
                    html_file_path=pathname2url(f"{gpxfile.stem}.html"),
                )
            )

    # Save the overview map as an HTML file
    html_file = "data/html/map.html"
    my_map.save(html_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
