import type { RouteDetail, RouteSummary } from "../types";

const base = import.meta.env.BASE_URL;

export async function fetchRoutes(): Promise<RouteSummary[]> {
  const response = await fetch(`${base}data/routes.json`);
  if (!response.ok) {
    throw new Error("Could not load routes");
  }
  return response.json() as Promise<RouteSummary[]>;
}

export async function fetchRoute(slug: string): Promise<RouteDetail> {
  const response = await fetch(`${base}data/routes/${slug}.json`);
  if (!response.ok) {
    throw new Error(`Could not load route ${slug}`);
  }
  return response.json() as Promise<RouteDetail>;
}

export function routeUrl(slug: string): string {
  return `${base}routes/${slug}/`;
}
