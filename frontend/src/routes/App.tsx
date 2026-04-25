import { createColumnHelper, flexRender, getCoreRowModel, useReactTable } from "@tanstack/react-table";
import { Bike, Download, ExternalLink, MapIcon, Share2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ElevationChart } from "../components/ElevationChart";
import { RouteIcon } from "../components/RouteIcon";
import { RouteMap } from "../components/RouteMap";
import { Stat } from "../components/Stat";
import { Button, ButtonLink } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { fetchRoute, fetchRoutes, routeUrl } from "../lib/data";
import { formatDistance, formatMeters } from "../lib/utils";
import type { RouteDetail, RouteSummary, TrackPoint } from "../types";

const directions = [
  ["NW", "N", "NE"],
  ["W", "C", "E"],
  ["SW", "S", "SE"],
];

const columnHelper = createColumnHelper<RouteSummary>();

export function App() {
  const [routes, setRoutes] = useState<RouteSummary[]>([]);
  const [selectedSlug, setSelectedSlug] = useState<string | null>(
    () => new URLSearchParams(window.location.search).get("route")
  );
  const [selectedRoute, setSelectedRoute] = useState<RouteDetail | null>(null);
  const [overviewRoutes, setOverviewRoutes] = useState<RouteDetail[]>([]);
  const [overviewOpen, setOverviewOpen] = useState(false);
  const [hoverPoint, setHoverPoint] = useState<TrackPoint | null>(null);

  useEffect(() => {
    fetchRoutes().then((nextRoutes) => {
      setRoutes(nextRoutes);
      if (!selectedSlug && nextRoutes[0]) {
        setSelectedSlug(nextRoutes[0].slug);
      }
    });
  }, [selectedSlug]);

  useEffect(() => {
    if (!overviewOpen || !routes.length || overviewRoutes.length) {
      return;
    }
    Promise.all(routes.map((route) => fetchRoute(route.slug))).then(setOverviewRoutes);
  }, [overviewOpen, overviewRoutes.length, routes]);

  useEffect(() => {
    if (!selectedSlug) {
      return;
    }
    fetchRoute(selectedSlug).then(setSelectedRoute);
    const url = new URL(window.location.href);
    url.searchParams.set("route", selectedSlug);
    window.history.replaceState(null, "", url);
  }, [selectedSlug]);

  const selectRoute = useCallback((slug: string) => {
    setSelectedSlug(slug);
    setOverviewOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const table = useReactTable({
    data: routes,
    columns: useMemo(
      () => [
        columnHelper.accessor("name", {
          header: "Route",
          cell: (info) => (
            <button
              className="flex items-center gap-2 text-left font-medium text-stone-950 hover:text-emerald-700"
              onClick={() => selectRoute(info.row.original.slug)}
            >
              <RouteIcon icon={info.row.original.icon} />
              {info.getValue()}
            </button>
          ),
        }),
        columnHelper.accessor("direction", { header: "Dir" }),
        columnHelper.accessor("distance", {
          header: "Distance",
          cell: (info) => formatDistance(info.getValue()),
        }),
        columnHelper.accessor("elevation_gain", {
          header: "Gain",
          cell: (info) => formatMeters(info.getValue()),
        }),
        columnHelper.accessor("climb_count", { header: "Climbs" }),
      ],
      [selectRoute]
    ),
    getCoreRowModel: getCoreRowModel(),
  });

  const byDirection = useMemo(() => {
    const grouped = new globalThis.Map<string, RouteSummary[]>();
    routes.forEach((route) => {
      const group = grouped.get(route.direction) ?? [];
      group.push(route);
      grouped.set(route.direction, group);
    });
    return grouped;
  }, [routes]);

  return (
    <main className="min-h-screen bg-stone-50 text-stone-900">
      <section className="border-b border-stone-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-emerald-700">
                <Bike className="h-4 w-4" />
                Leuven GPX
              </div>
              <h1 className="mt-2 text-3xl font-semibold tracking-normal text-stone-950">
                Race bike routes from Leuven
              </h1>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button onClick={() => setOverviewOpen((value) => !value)}>
                <MapIcon className="h-4 w-4" />
                Overview map
              </Button>
              {selectedRoute ? (
                <ButtonLink href={selectedRoute.summary.gpx_url} download>
                  <Download className="h-4 w-4" />
                  GPX
                </ButtonLink>
              ) : null}
            </div>
          </div>

          {selectedRoute ? (
            <div className="grid gap-3 sm:grid-cols-4">
              <Stat label="Distance" value={formatDistance(selectedRoute.summary.distance)} />
              <Stat label="Elevation" value={formatMeters(selectedRoute.summary.elevation_gain)} />
              <Stat label="Climbs" value={String(selectedRoute.summary.climb_count)} />
              <Stat label="Direction" value={selectedRoute.summary.direction} />
            </div>
          ) : null}
        </div>
      </section>

      <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[360px_1fr] lg:px-8">
        <aside className="space-y-5">
          <Card className="p-4">
            <h2 className="text-base font-semibold text-stone-950">Directions</h2>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {directions.flat().map((direction) => (
                <div key={direction} className="rounded-md border border-stone-200 bg-stone-50 p-2">
                  <div className="text-xs font-semibold text-stone-500">{direction}</div>
                  <div className="mt-1 space-y-1">
                    {(byDirection.get(direction) ?? []).map((route) => (
                      <button
                        key={route.slug}
                        title={`${formatDistance(route.distance)}, ${formatMeters(route.elevation_gain)}, ${route.climb_count} climbs`}
                        onClick={() => selectRoute(route.slug)}
                        className="block w-full truncate rounded-sm px-1 py-0.5 text-left text-sm hover:bg-white hover:text-emerald-700"
                      >
                        {route.name}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card className="overflow-hidden">
            <div className="border-b border-stone-200 p-4">
              <h2 className="text-base font-semibold text-stone-950">All routes</h2>
            </div>
            <div className="max-h-[560px] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-stone-100 text-xs uppercase text-stone-500">
                  {table.getHeaderGroups().map((headerGroup) => (
                    <tr key={headerGroup.id}>
                      {headerGroup.headers.map((header) => (
                        <th key={header.id} className="px-3 py-2 text-left font-semibold">
                          {flexRender(header.column.columnDef.header, header.getContext())}
                        </th>
                      ))}
                    </tr>
                  ))}
                </thead>
                <tbody>
                  {table.getRowModel().rows.map((row) => (
                    <tr key={row.id} className="border-t border-stone-100">
                      {row.getVisibleCells().map((cell) => (
                        <td key={cell.id} className="px-3 py-2 align-top">
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </aside>

        <section className="space-y-6">
          {overviewOpen ? (
            <Card className="p-3">
              <div className="h-[520px]">
                <RouteMap
                  points={[]}
                  summaries={routes}
                  overviewRoutes={overviewRoutes}
                  onRouteSelect={selectRoute}
                />
              </div>
            </Card>
          ) : null}

          {selectedRoute ? (
            <>
              <Card className="overflow-hidden">
                <div className="flex flex-col gap-3 border-b border-stone-200 p-4 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h2 className="text-2xl font-semibold text-stone-950">
                      {selectedRoute.summary.name}
                    </h2>
                    <p className="text-sm text-stone-500">{selectedRoute.summary.file_name}</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <ButtonLink href={routeUrl(selectedRoute.summary.slug)}>
                      <Share2 className="h-4 w-4" />
                      Share
                    </ButtonLink>
                    <ButtonLink href={selectedRoute.summary.gpx_url} download>
                      <Download className="h-4 w-4" />
                      Download
                    </ButtonLink>
                  </div>
                </div>
                <div className="h-[520px] p-3">
                  <RouteMap
                    points={selectedRoute.points}
                    climbs={selectedRoute.climbs}
                    hoverPoint={hoverPoint}
                  />
                </div>
              </Card>

              <ElevationChart
                points={selectedRoute.points}
                climbs={selectedRoute.climbs}
                onHoverPoint={setHoverPoint}
              />

              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {selectedRoute.climbs.map((climb) => (
                  <Card key={climb.id} className="p-4">
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="text-lg font-semibold text-stone-950">Climb {climb.id}</h3>
                      <span className="rounded-full bg-rose-100 px-2 py-1 text-xs font-medium text-rose-700">
                        Cat {climb.category}
                      </span>
                    </div>
                    <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                      <Stat label="Length" value={formatDistance(climb.length)} />
                      <Stat label="Ascent" value={formatMeters(climb.ascent)} />
                      <Stat label="Average" value={`${climb.grade.toFixed(1)}%`} />
                      <Stat label="Max" value={`${climb.max_grade.toFixed(1)}%`} />
                    </div>
                  </Card>
                ))}
              </div>

              <p className="flex items-center gap-2 text-sm text-stone-500">
                <ExternalLink className="h-4 w-4" />
                Map tiles by OpenStreetMap. Climbs are detected automatically from the GPX elevation profile.
              </p>
            </>
          ) : (
            <Card className="p-6">Loading routes...</Card>
          )}
        </section>
      </div>
    </main>
  );
}
