import sys
from pathlib import Path
from typing import List
from urllib.request import pathname2url
from xml.dom import minidom

import altair as alt
import folium
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# from scipy.signal import find_peaks


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
        ave_lat, ave_lon, lon_list, lat_list, h_list = get_gpx(gpxfile)

        color = "red" if gpxfile.name == "DR SW Huldenberg.gpx" else "#38b580"
        # Add a polyline to connect the track points
        folium.PolyLine(
            list(zip(lat_list, lon_list)), color=color, weight=2.5, opacity=0.8
        ).add_to(my_map)

        route_map = folium.Map(location=[ave_lat, ave_lon], zoom_start=12)
        folium.PolyLine(
            list(zip(lat_list, lon_list)), color="red", weight=2.5, opacity=1
        ).add_to(route_map)
        html_file = f"data/html/{gpxfile.stem}.html"
        route_map.save(html_file)

        df = pd.DataFrame({"lon": lon_list, "lat": lat_list, "elev": h_list})
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
        elevation = (
            alt.Chart(df)
            .mark_area(
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="lightgrey", offset=0),
                        alt.GradientStop(color="darkgrey", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
                line={"color": "darkgreen"},
            )
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
            )
            .properties(width=800)
            .configure_view(
                strokeWidth=0,
            )
        )
        json_file_path = f"data/html/{gpxfile.stem}.json"
        with open(json_file_path, "w") as json_file:
            json_file.write(elevation.to_json())
        profile_template = env.get_template("profile.html")
        with open(f"data/html/route_{gpxfile.stem}.html", "w") as fh:
            fh.write(
                profile_template.render(
                    json_file_path=pathname2url(json_file_path),
                )
            )

    # Save the map as an HTML file
    html_file = "data/html/map.html"
    my_map.save(html_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
