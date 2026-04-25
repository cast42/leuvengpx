export type LatLon = [number, number];

export type RouteSummary = {
  slug: string;
  name: string;
  file_name: string;
  direction: string;
  distance: number;
  elevation_gain: number;
  climb_count: number;
  start: LatLon;
  end: LatLon;
  bounds: [LatLon, LatLon];
  detail_url: string;
  gpx_url: string;
  icon: string;
};

export type TrackPoint = {
  lat: number;
  lon: number;
  elevation: number;
  distance: number;
};

export type Climb = {
  id: number;
  start_index: number;
  end_index: number;
  start_distance: number;
  end_distance: number;
  length: number;
  ascent: number;
  grade: number;
  score: number;
  category: number;
  lat: number;
  lon: number;
  max_grade: number;
};

export type RouteDetail = {
  summary: RouteSummary;
  points: TrackPoint[];
  climbs: Climb[];
};
