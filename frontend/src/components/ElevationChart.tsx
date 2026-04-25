import { useMemo, useRef, useState } from "react";

import type { Climb, TrackPoint } from "../types";

type Props = {
  points: TrackPoint[];
  climbs: Climb[];
  onHoverPoint: (point: TrackPoint | null) => void;
};

const width = 960;
const height = 240;
const padding = { top: 14, right: 18, bottom: 32, left: 44 };

const categoryColors = new Map<number, string>([
  [0, "#7f1d1d"],
  [1, "#dc2626"],
  [2, "#f97316"],
  [3, "#facc15"],
  [4, "#4ade80"],
  [5, "#4ade80"],
]);

function tickStep(distance: number): number {
  if (distance <= 50) {
    return 5;
  }
  if (distance <= 100) {
    return 10;
  }
  if (distance <= 200) {
    return 25;
  }
  return 50;
}

export function ElevationChart({ points, climbs, onHoverPoint }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<{ point: TrackPoint; index: number } | null>(null);
  const model = useMemo(() => {
    const minElevation = Math.min(...points.map((point) => point.elevation));
    const maxElevation = Math.max(...points.map((point) => point.elevation));
    const maxDistance = points[points.length - 1]?.distance ?? 1;
    const x = (distance: number) =>
      padding.left +
      (distance / maxDistance) * (width - padding.left - padding.right);
    const y = (elevation: number) =>
      height -
      padding.bottom -
      ((elevation - minElevation) / Math.max(maxElevation - minElevation, 1)) *
        (height - padding.top - padding.bottom);
    const linePath = points
      .map((point, index) => `${index === 0 ? "M" : "L"} ${x(point.distance)} ${y(point.elevation)}`)
      .join(" ");
    const baseY = height - padding.bottom;
    const areaPath = `${linePath} L ${width - padding.right} ${baseY} L ${padding.left} ${baseY} Z`;
    const ticks = [];
    const step = tickStep(maxDistance);
    for (let tick = 0; tick <= maxDistance; tick += step) {
      ticks.push(tick);
    }
    if (maxDistance - ticks[ticks.length - 1] > step * 0.35) {
      ticks.push(maxDistance);
    } else {
      ticks[ticks.length - 1] = maxDistance;
    }
    return { minElevation, maxElevation, maxDistance, x, y, linePath, areaPath, ticks };
  }, [points]);

  const hoverTooltip = useMemo(() => {
    if (!hover) {
      return null;
    }
    const previous = points[Math.max(0, hover.index - 1)];
    const next = points[Math.min(points.length - 1, hover.index + 1)];
    const distanceMeters = Math.max((next.distance - previous.distance) * 1000, 1);
    const grade = ((next.elevation - previous.elevation) / distanceMeters) * 100;
    const x = model.x(hover.point.distance);
    const y = model.y(hover.point.elevation);
    const tooltipWidth = 130;
    const tooltipHeight = 58;
    const tooltipX =
      x > width - padding.right - tooltipWidth - 18 ? x - tooltipWidth - 10 : x + 10;
    const tooltipY = Math.max(
      padding.top + 4,
      Math.min(y - tooltipHeight / 2, height - padding.bottom - tooltipHeight - 4)
    );
    return {
      grade,
      x,
      y,
      tooltipX,
      tooltipY,
      tooltipWidth,
      tooltipHeight,
    };
  }, [hover, model, points]);

  const climbAreas = useMemo(
    () =>
      climbs
        .map((climb) => {
          const climbPoints = points.slice(climb.start_index, climb.end_index + 1);
          if (climbPoints.length < 2) {
            return null;
          }
          const top = climbPoints
            .map(
              (point, index) =>
                `${index === 0 ? "M" : "L"} ${model.x(point.distance)} ${model.y(point.elevation)}`
            )
            .join(" ");
          const start = climbPoints[0];
          const end = climbPoints[climbPoints.length - 1];
          return {
            id: climb.id,
            color: categoryColors.get(climb.category) ?? "#94a3b8",
            path: `${top} L ${model.x(end.distance)} ${height - padding.bottom} L ${model.x(start.distance)} ${height - padding.bottom} Z`,
          };
        })
        .filter((area): area is { id: number; color: string; path: string } => Boolean(area)),
    [climbs, model, points]
  );

  function handlePointerMove(event: React.PointerEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    const localX = ((event.clientX - rect.left) / rect.width) * width;
    const distance =
      ((localX - padding.left) / (width - padding.left - padding.right)) *
      model.maxDistance;
    let closestIndex = 0;
    for (let index = 1; index < points.length; index += 1) {
      if (
        Math.abs(points[index].distance - distance) <
        Math.abs(points[closestIndex].distance - distance)
      ) {
        closestIndex = index;
      }
    }
    const point = points[closestIndex];
    setHover({ point, index: closestIndex });
    onHoverPoint(point);
  }

  return (
    <svg
      ref={svgRef}
      className="h-64 w-full touch-none rounded-lg border border-stone-200 bg-white"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Elevation profile"
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerMove}
      onPointerLeave={() => {
        setHover(null);
        onHoverPoint(null);
      }}
    >
      {[0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = padding.top + (height - padding.top - padding.bottom) * ratio;
        return (
          <line
            key={ratio}
            x1={padding.left}
            x2={width - padding.right}
            y1={y}
            y2={y}
            stroke="#e7e5e4"
            strokeDasharray="4 4"
          />
        );
      })}
      <line
        x1={padding.left}
        x2={width - padding.right}
        y1={height - padding.bottom}
        y2={height - padding.bottom}
        stroke="#d6d3d1"
      />
      <line
        x1={padding.left}
        x2={padding.left}
        y1={padding.top}
        y2={height - padding.bottom}
        stroke="#d6d3d1"
      />
      <path
        d={model.areaPath}
        fill="#ccfbf1"
        opacity="0.7"
      />
      {climbAreas.map((area) => (
        <path key={area.id} d={area.path} fill={area.color} opacity="0.72" />
      ))}
      <path d={model.linePath} fill="none" stroke="#0f766e" strokeWidth="3" />
      {climbs.map((climb) => (
        <g key={climb.id}>
          <line
            x1={model.x(climb.start_distance)}
            x2={model.x(climb.end_distance)}
            y1={height - padding.bottom + 4}
            y2={height - padding.bottom + 4}
            stroke="#e11d48"
            strokeWidth="5"
          />
          <text
            x={model.x(climb.end_distance)}
            y={model.y(points[climb.end_index]?.elevation ?? model.maxElevation) - 8}
            textAnchor="middle"
            className="fill-rose-700 text-xs font-semibold"
          >
            {climb.id}
          </text>
        </g>
      ))}
      {model.ticks.map((tick) => (
        <g key={tick}>
          <line
            x1={model.x(tick)}
            x2={model.x(tick)}
            y1={height - padding.bottom}
            y2={height - padding.bottom + 5}
            stroke="#a8a29e"
          />
          <text
            x={model.x(tick)}
            y={height - 9}
            textAnchor={tick === 0 ? "start" : tick === model.maxDistance ? "end" : "middle"}
            className="fill-stone-500 text-xs"
          >
            {tick === model.maxDistance ? `${tick.toFixed(1)} km` : `${Math.round(tick)} km`}
          </text>
        </g>
      ))}
      <rect
        x={padding.left}
        y={padding.top}
        width={width - padding.left - padding.right}
        height={height - padding.top - padding.bottom}
        fill="transparent"
        pointerEvents="all"
      />
      {hoverTooltip && hover ? (
        <g>
          <line
            x1={hoverTooltip.x}
            x2={hoverTooltip.x}
            y1={padding.top}
            y2={height - padding.bottom}
            stroke="#57534e"
            strokeWidth="1.5"
          />
          <circle
            cx={hoverTooltip.x}
            cy={hoverTooltip.y}
            r="5"
            fill="#facc15"
            stroke="#1c1917"
            strokeWidth="2"
          />
          <g className="pointer-events-none">
            <rect
              x={hoverTooltip.tooltipX}
              y={hoverTooltip.tooltipY}
              width={hoverTooltip.tooltipWidth}
              height={hoverTooltip.tooltipHeight}
              fill="white"
              stroke="#78716c"
              strokeWidth="1"
              opacity="0.96"
            />
            <text
              x={hoverTooltip.tooltipX + 8}
              y={hoverTooltip.tooltipY + 17}
              className="fill-stone-700 text-xs"
            >
              Distance
              <tspan dx="8" className="font-semibold fill-stone-950">
                {hover.point.distance.toFixed(2)} km
              </tspan>
            </text>
            <text
              x={hoverTooltip.tooltipX + 8}
              y={hoverTooltip.tooltipY + 34}
              className="fill-stone-700 text-xs"
            >
              Elevation
              <tspan dx="8" className="font-semibold fill-stone-950">
                {Math.round(hover.point.elevation)} m
              </tspan>
            </text>
            <text
              x={hoverTooltip.tooltipX + 8}
              y={hoverTooltip.tooltipY + 51}
              className="fill-stone-700 text-xs"
            >
              Grade
              <tspan dx="8" className="font-semibold fill-stone-950">
                {hoverTooltip.grade.toFixed(1)}%
              </tspan>
            </text>
          </g>
        </g>
      ) : null}
      <text x={8} y={model.y(model.maxElevation)} className="fill-stone-500 text-xs">
        {Math.round(model.maxElevation)} m
      </text>
      <text x={8} y={model.y(model.minElevation)} className="fill-stone-500 text-xs">
        {Math.round(model.minElevation)} m
      </text>
    </svg>
  );
}
