from pathlib import Path
from urllib.request import pathname2url
from xml.dom import minidom

import folium
from jinja2 import Environment, FileSystemLoader

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
        lon, lat = track[s].attributes["lon"].value, track[s].attributes["lat"].value
        elev = elevation[s].firstChild.nodeValue
        lon_list.append(float(lon))
        lat_list.append(float(lat))
        h_list.append(float(elev))

    # Calculate average latitude and longitude
    ave_lat = sum(lat_list) / len(lat_list)
    ave_lon = sum(lon_list) / len(lon_list)

    return ave_lat, ave_lon, lon_list, lat_list, h_list


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

# Save the map as an HTML file
html_file = "data/html/map.html"
my_map.save(html_file)
