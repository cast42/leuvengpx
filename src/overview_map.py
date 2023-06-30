from pathlib import Path
from xml.dom import minidom

import folium

for gpxfile in Path("/Users/lode/projects/leuvengpx/data/gpx/").glob("*.gpx"):
    print(gpxfile.as_posix())


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
    # ave_lat = sum(lat_list) / len(lat_list)
    # ave_lon = sum(lon_list) / len(lon_list)

    return lon_list, lat_list, h_list


# Create a folium map
my_map = folium.Map(location=[50.876777, 4.715101], zoom_start=10)

for gpxfile in Path("/Users/lode/projects/leuvengpx/data/gpx/").glob("*.gpx"):
    print(gpxfile.as_posix())
    lon_list, lat_list, h_list = get_gpx(gpxfile)

    # Add a polyline to connect the track points
    folium.PolyLine(
        list(zip(lat_list, lon_list)), color="#38b580", weight=1.5, opacity=0.8
    ).add_to(my_map)

# Save the map as an HTML file
html_file = "/Users/lode/projects/leuvengpx/data/html/map.html"
my_map.save(html_file)
