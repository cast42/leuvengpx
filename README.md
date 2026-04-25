# Leuven GPX

Static website for race bike GPX routes that start in Leuven.

Maintainers add GPX files to `data/gpx`. The Python generator reads those files,
calculates distance, elevation gain and climbs, and writes JSON plus share-preview
HTML into `public/`. The React frontend renders the homepage, overview map, route
detail pages, elevation chart, climb cards and GPX download links.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+
- [just](https://github.com/casey/just)

## Setup

```bash
just install
```

## Common Commands

```bash
just generate            # Generate public route data from data/gpx
just preview             # Generate data and start the local Vite dev server
just build               # Generate data and build dist/
just serve               # Build and serve dist/ locally
just add-gpx path/to.gpx # Copy a GPX into data/gpx and regenerate data
just check               # Python lint, typing, tests, frontend typecheck and build
```

The expected GPX filename format is:

```text
DR <direction> <route name>.gpx
```

Valid directions are `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW` and `C`.

## Publishing

`just publish` runs the full check locally, then pushes the current commit to
`origin/main`. That push triggers the `pages` GitHub Actions workflow, which
builds `dist/` and deploys it with GitHub Pages.

Configure GitHub Pages to use GitHub Actions as its source. The Vite base path is
`/leuvengpx/`, matching `https://cast42.github.io/leuvengpx/`.

## Route Sharing

Every route gets a generated preview page at:

```text
routes/<route-slug>/
```

Those pages contain route-specific Open Graph metadata for Slack, WhatsApp and
other unfurlers, then redirect visitors into the interactive React route detail.

## Climb Detection

Climbs are detected from cleaned GPX elevation data:

1. A median filter removes sharp elevation spikes.
2. Local valleys and peaks are paired into candidate climbs.
3. Candidates are kept when their climb score is at least `1500` and length is
   at least `250 m`.

The climb score follows the Garmin-style formula:

```text
length_in_meters * average_grade_percent
```
