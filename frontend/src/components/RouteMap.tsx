import L, { type LatLngExpression, type Map as LeafletMap } from "leaflet";
import { useEffect, useMemo, useRef } from "react";

import type { Climb, RouteSummary, TrackPoint } from "../types";
import { formatDistance, formatMeters } from "../lib/utils";

type RouteMapProps = {
  points: TrackPoint[];
  climbs?: Climb[];
  summaries?: RouteSummary[];
  overviewRoutes?: { summary: RouteSummary; points: TrackPoint[] }[];
  hoverPoint?: TrackPoint | null;
  onRouteSelect?: (slug: string) => void;
};

const startIcon = L.divIcon({
  className: "map-pin map-pin-start",
  html: "S",
  iconSize: [26, 26],
  iconAnchor: [13, 13],
});

const endIcon = L.divIcon({
  className: "map-pin map-pin-end",
  html: "E",
  iconSize: [26, 26],
  iconAnchor: [13, 13],
});

function climbIcon(id: number) {
  return L.divIcon({
    className: "map-pin map-pin-climb",
    html: String(id),
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  });
}

export function RouteMap({
  points,
  climbs = [],
  summaries,
  overviewRoutes,
  hoverPoint,
  onRouteSelect,
}: RouteMapProps) {
  const elementRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const hoverMarkerRef = useRef<L.CircleMarker | null>(null);

  const positions = useMemo<LatLngExpression[]>(
    () => points.map((point) => [point.lat, point.lon]),
    [points]
  );

  useEffect(() => {
    if (!elementRef.current || mapRef.current) {
      return;
    }
    mapRef.current = L.map(elementRef.current, {
      scrollWheelZoom: false,
      preferCanvas: true,
    });
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(mapRef.current);
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }

    const layer = L.layerGroup().addTo(map);
    if (summaries) {
      const allBounds: LatLngExpression[] = [];
      summaries.forEach((route) => {
        const routePoints =
          overviewRoutes?.find((detail) => detail.summary.slug === route.slug)?.points ?? [];
        const routePositions = routePoints.map((point) => [point.lat, point.lon] as LatLngExpression);
        allBounds.push(...(routePositions.length ? routePositions : [route.bounds[0], route.bounds[1]]));
        const line = L.polyline(routePositions, {
          color: "#0f766e",
          weight: 3,
          opacity: 0.72,
        });
        line.bindPopup(
          `<strong>${route.name}</strong><br>${formatDistance(route.distance)}<br>${formatMeters(route.elevation_gain)} gain<br>${route.climb_count} climbs<br><button class="popup-link" data-route="${route.slug}">Open route</button>`
        );
        line.on("popupopen", () => {
          const button = document.querySelector(`[data-route="${route.slug}"]`);
          button?.addEventListener("click", () => onRouteSelect?.(route.slug), {
            once: true,
          });
        });
        line.addTo(layer);
      });
      if (allBounds.length) {
        map.fitBounds(L.latLngBounds(allBounds), { padding: [24, 24] });
      }
    } else if (positions.length) {
      L.polyline(positions, {
        color: "#e11d48",
        weight: 4,
        opacity: 0.95,
      }).addTo(layer);
      L.polyline(positions, { color: "white", weight: 1.4, opacity: 0.9 }).addTo(layer);
      L.marker(positions[0], { icon: startIcon }).addTo(layer);
      L.marker(positions[positions.length - 1], { icon: endIcon }).addTo(layer);
      climbs.forEach((climb) => {
        L.marker([climb.lat, climb.lon], { icon: climbIcon(climb.id) })
          .bindPopup(
            `<strong>Climb ${climb.id}</strong><br>${formatDistance(climb.length)}<br>${formatMeters(climb.ascent)} gain<br>${climb.grade.toFixed(1)}% avg`
          )
          .addTo(layer);
      });
      map.fitBounds(L.latLngBounds(positions), { padding: [24, 24] });
    }

    return () => {
      layer.remove();
    };
  }, [climbs, onRouteSelect, overviewRoutes, positions, summaries]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) {
      return;
    }
    if (!hoverPoint) {
      hoverMarkerRef.current?.remove();
      hoverMarkerRef.current = null;
      return;
    }
    const latLng: LatLngExpression = [hoverPoint.lat, hoverPoint.lon];
    if (!hoverMarkerRef.current) {
      hoverMarkerRef.current = L.circleMarker(latLng, {
        radius: 8,
        color: "#111827",
        fillColor: "#facc15",
        fillOpacity: 1,
        weight: 2,
      }).addTo(map);
    } else {
      hoverMarkerRef.current.setLatLng(latLng);
    }
  }, [hoverPoint]);

  return <div ref={elementRef} className="h-full min-h-80 w-full rounded-lg" />;
}
