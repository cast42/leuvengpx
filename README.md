## Race bike GPX files starting in Leuven Belgium

This repository contains a Python script to generate HTML starting from a folder of GPX files.
The result is an interactive overview map, containing all gpx routes and a table with links
to detailed overviews per route plus a download button for the GPX files.

The detail page per route shows a map with the route and a height profile showing
the altitude and climbs along the route.

The climbs are detected using scipy's [find_peak](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) algorithm.

The GPX files must be stored in the data/gpx folder.
All files matching data/gpx/*.gpx will appear on the front page index.html.
The GPX files must comply to the following template: `r"^DR\s[N|E|W|C|S]{1,2}\s[\w\s]+\.gpx$"`

This project was derived from the [minimal python boilerplate](https://github.com/datarootsio/python-minimal-boilerplate). The template project contains the following setup ready for you to go:

* package/environment management
  * `poetry`
* code validation
  * `black`
  * `ruff`
  * `pytest`
* pre-commit hooks

# Setup

1. Clone this repo
2. Make sure to [install Poetry](https://python-poetry.org/docs/#installation)
3. In your project directory run
   1. `poetry shell`
   2. `poetry install`
   3. `pre-commit install`
4. On each `git commit` the code validation packages will be run before the actual commit.
5. Add your GPX files to the data/gpx folder
6. Run `python generate_html_from_gpx.py`to generate index.html and content of data/html
7. Goto Pages in Settings to make this a Github Pages repository.
8. Browse to `https://<username>.github.io/<repository>`. E.g. [https://cast42.github.io/leuvengpx/](https://cast42.github.io/leuvengpx/)

# Adding a new gpx file

From the root of this project, add the gpx file "DR C new.gpx" to the data/gpx directory, generate the new version of the site by running the python script and check-in the new files into the version control:

1. `cp DR\ C\ new.gpx data/gpx`
2. `python src/generate_html_from_gpx.py`
3. `git add  data/gpx/DR\ C\ new.gpx`
4. `git add data/html/*.html`
5. `git commit -m "added new.gpx"`
6. `git push`
