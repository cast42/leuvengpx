import json
import logging
import sys
from collections import defaultdict
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
from folium.features import DivIcon
from jinja2 import Environment, FileSystemLoader
from scipy.signal import find_peaks

START_LOCATION = [50.876777, 4.715101]  # Position dataroots office
GPX_FILE_TO_HIGHLIGHT = (
    "DR NE De vogel-lokker.gpx"  # Filename of gpx to highlight in red
)
APPLY_GARMIN_RULES = False  # Filter climbs with Garmin rules


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
        .assign(max_elevation=df["elev"].max().round(-1) + 10)
    )
    # Garmin rules
    if APPLY_GARMIN_RULES:
        df_peaks_filtered = df_peaks_filtered.query(
            "(climb_score >= 1_500) & (length >= 0.5) & (grade >= 3_000)"
        )
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
            df[
                [
                    "distance_from_start",
                    "smoothed_elevation",
                    "smoothed_grade_color",
                    "grade",
                ]
            ]
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
            tooltip=[
                alt.Tooltip(
                    "distance_from_start:Q", title="Distance (km)", format=".2f"
                ),
                alt.Tooltip("smoothed_elevation:Q", title="Elevation (m)", format="d"),
                alt.Tooltip("grade_percent:Q", title="Grade (%)", format=".0%"),
            ],
        )
        .transform_calculate(
            grade_percent="datum.grade/100",
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

    df_annot = (
        df_peaks_filtered.reset_index(drop=True)
        .assign(number=lambda df_: df_.index + 1)
        .assign(circle_pos=lambda df_: df_["max_elevation"] + 20)[
            [
                "distance_from_start",
                "max_elevation",
                "circle_pos",
                "number",
                "length",
                "total_ascent",
                "grade",
                "climb_score",
                "prev_distance_from_start",
            ]
        ]
    )

    annotation = (
        alt.Chart(df_annot)
        .mark_text(align="center", baseline="bottom", fontSize=16, dy=-10)
        .encode(
            x=alt.X(
                "distance_from_start:Q",
                scale=alt.Scale(domain=(0, total_distance_round)),
            ),
            y="max_elevation",
            text="number",
            tooltip=[
                alt.Tooltip(
                    "prev_distance_from_start:Q", title="Starts at (km)", format=".2f"
                ),
                alt.Tooltip("total_ascent:Q", title="Total ascent (m)", format="d"),
                alt.Tooltip("length:Q", title="Length (km)", format=".2f"),
                alt.Tooltip("grade_percent:Q", title="Average Grade", format=".0%"),
                alt.Tooltip("climb_score:Q", title="Climb score", format="d"),
            ],
        )
        .transform_calculate(
            grade_percent="datum.grade/(100*1000)",
            max_grade_percent="datum.max_grade/(100*1000)",
        )
    )
    chart = (
        (elevation + line_peaks + annotation)
        .properties(width="container")
        .configure_view(
            strokeWidth=0,
        )
    )
    return chart, df_peaks_filtered


def generate_index_html():
    templates_dir = "data/templates"
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("index.html")

    direction_gpx_files = defaultdict(list)
    all_wind_directions = ["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"]
    for gpxfile in Path("data/gpx/").glob("*.gpx"):
        wind_direction = gpxfile.stem.split()[1]
        if wind_direction not in all_wind_directions:
            logging.warning(f"Found gpx file {gpxfile.stem} has no wind direction.")
            logging.warning("Please add a winddirection to filename of gpx file.")
            logging.warning(
                "Possible wind directions are: {','.join(all_wind_directions)}"
            )
            logging.warning("Skipping this file")
        direction_gpx_files[wind_direction].append(gpxfile)

    wind_rows = [["NW", "N", "NE"], ["W", "C", "E"], ["SW", "S", "SE"]]
    rows = []
    for wind_row in wind_rows:
        row = []
        for wind_direction in wind_row:
            gpx_files_in_this_direction = direction_gpx_files[wind_direction]
            table_element = []
            if len(gpx_files_in_this_direction):
                for gpxfile in gpx_files_in_this_direction:
                    table_element.append(
                        {
                            "html_file": pathname2url(gpxfile.stem) + ".html",
                            "gpx_file": pathname2url(gpxfile.name),
                            "gpx_name": gpxfile.name,
                        }
                    )
            else:
                table_element.append({})
            row.append(table_element)
        rows.append(row)

    with open("index.html", "w") as fh:
        fh.write(template.render(rows=rows))


def generate_overview_map():
    # Create a folium map
    my_map = folium.Map(location=START_LOCATION, zoom_start=11)
    for gpxfile in Path("data/gpx/").glob("*.gpx"):
        # Open gpx file and parse its content
        ave_lat, ave_lon, lon_list, lat_list, h_list = get_gpx(gpxfile)
        color = "red" if gpxfile.name == GPX_FILE_TO_HIGHLIGHT else "#38b580"
        # Add a polyline to connect the track points
        folium.PolyLine(
            list(zip(lat_list, lon_list)),
            color=color,
            fillColor="white",
            weight=5,
            opacity=1,
        ).add_to(my_map)
        folium.PolyLine(
            list(zip(lat_list, lon_list)),
            color="white",
            weight=1,
            opacity=1,
        ).add_to(my_map)
    # Save the overview map as an HTML file
    html_file = "data/html/map.html"
    my_map.save(html_file)


def generate_climb_profile(df_hill: pd.DataFrame, title: str):
    climb_profile = (
        alt.Chart(
            df_hill,
            title=alt.Title(
                title,
                anchor="start",
            ),
        )
        .mark_area()
        .encode(
            x=alt.X("distance_from_start")
            .axis(grid=False, tickCount=10, labelExpr="datum.label + ' m'", title=None)
            .scale(domain=(0, df_hill["distance_from_start"].max())),
            y=alt.Y("elev").axis(
                domain=False,
                ticks=False,
                tickCount=5,
                labelExpr="datum.label + ' m'",
                title=None,
            ),
            color=alt.Color("color_grade").scale(None),
            tooltip=[
                alt.Tooltip("distance_from_start:Q", title="Distance (m)", format="d"),
                alt.Tooltip("elev:Q", title="Elevation (m)", format="d"),
                alt.Tooltip("grade_percent:Q", title="Grade (%)", format=".0%"),
            ],
        )
        .transform_calculate(
            grade_percent="datum.grade/100",
        )
    )
    return climb_profile


def generate_page_per_route():
    for gpxfile in Path("data/gpx/").glob("*.gpx"):
        # Open gpx file and parse its content
        ave_lat, ave_lon, lon_list, lat_list, h_list = get_gpx(gpxfile)
        # Store coordinates and elevation in a Pandas dataframe
        df = pd.DataFrame({"lon": lon_list, "lat": lat_list, "elev": h_list})

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

        chart, df_peaks = generate_height_profile_json(df)
        height_profile = features.VegaLite(
            json.loads(chart.to_json()),
            position="absolute",
            left="0%",
            width="100%",
            height="100%",
            top="70%",
        )

        for index, row in df_peaks.reset_index(drop=True).iterrows():
            icon_div = DivIcon(
                icon_size=(150, 36),
                icon_anchor=(7, 20),
                html=f"<div style='font-size: 18pt; color : black'>{index+1}</div>",
            )
            length = (
                f"{row['length']:.1f} km"
                if row["length"] >= 1
                else f"{row['length']*1000:.0f} m"
            )
            popup_text = f"""Climb {index+1}<br>
                    Lenght: {length}<br>
                    Avg. grade: {row['grade']/1000:.1f}%<br>
                    Total ascent: {int(row['total_ascent'])}m
            """
            popup = folium.Popup(popup_text, min_width=100, max_width=200)
            folium.Marker(
                [row["lat"], row["lon"]],
                popup=popup,
                icon=icon_div,
            ).add_to(route_map)
            circle = folium.CircleMarker(
                radius=15,
                location=[row["lat"], row["lon"]],
                # tooltip = label,
                color="crimson",
                fill=True,
            )
            circle.add_to(route_map)

        f.add_child(route_map)
        f.add_child(height_profile)

        div = branca.element.Div(width="100%")
        div.add_child(route_map)
        div.add_child(height_profile)

        f.add_child(div)

        for index, row in df_peaks.reset_index(drop=True).iterrows():
            df_hill = (
                df[
                    df["distance_from_start"].between(
                        row["prev_distance_from_start"],
                        row["distance_from_start"],
                    )
                ]
                .assign(
                    distance_from_start=lambda df_: (
                        df_["distance_from_start"] - row["prev_distance_from_start"]
                    )
                    * 1_000
                )
                .assign(color_grade=lambda df_: df_["grade"].map(grade_to_color))
            )
            df_new_index = pd.DataFrame(
                index=pd.Index(np.arange(0, df_hill["distance_from_start"].max(), 10))
            )
            df_hill_resample = (
                pd.concat(
                    [df_hill.set_index("distance_from_start"), df_new_index], axis=0
                )
                .sort_index()
                .assign(grade=lambda df_: df_["grade"].clip(lower=-35, upper=35))
            )
            df_hill_resample = (
                df_hill_resample[["elev", "grade"]]
                .interpolate(axis=0)
                .fillna(method="ffill", axis=0)
                .fillna(method="bfill", axis=0)
            )

            df_hill_resample["color_grade"] = df_hill_resample["grade"].map(
                grade_to_color
            )
            df_hill_resample = (
                df_hill_resample.reset_index()
                .rename(columns={"index": "distance_from_start"})
                .sort_values(by="distance_from_start")
            )
            max_grade = df_hill_resample["grade"].rolling(10).mean().max() / 100
            title = (
                f"""Climb {index+1}: {row['length']:.2f}km """
                f"""Total ascent: {int(row['total_ascent']):d}hm """
                f"""Grade: {(row['grade']/100_000):.1%} avg., """
                f"""{max_grade:.1%} max"""
            )
            altair_climb_profile = generate_climb_profile(df_hill_resample, title)
            folium_climb_profile = features.VegaLite(
                json.loads(altair_climb_profile.to_json()), height="30%"
            )
            f.add_child(folium_climb_profile)
        html_file = f"data/html/{gpxfile.stem}.html"
        f.save(html_file)


def main(cli_args: List[str] = None) -> int:
    logging.info("Generating index.html")
    generate_index_html()
    logging.info("Generating overview map with all gpx routes")
    generate_overview_map()
    logging.info("Generating page per route")
    generate_page_per_route()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
