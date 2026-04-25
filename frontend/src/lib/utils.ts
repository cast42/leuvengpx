import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDistance(value: number): string {
  return `${value.toFixed(1)} km`;
}

export function formatMeters(value: number): string {
  return `${Math.round(value)} m`;
}
