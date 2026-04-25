import {
  ArrowDown,
  ArrowDownLeft,
  ArrowDownRight,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpLeft,
  ArrowUpRight,
  CircleDot,
} from "lucide-react";

const icons = {
  "arrow-up": ArrowUp,
  "arrow-up-right": ArrowUpRight,
  "arrow-right": ArrowRight,
  "arrow-down-right": ArrowDownRight,
  "arrow-down": ArrowDown,
  "arrow-down-left": ArrowDownLeft,
  "arrow-left": ArrowLeft,
  "arrow-up-left": ArrowUpLeft,
  "circle-dot": CircleDot,
};

export function RouteIcon({ icon }: { icon: string }) {
  const Icon = icons[icon as keyof typeof icons] ?? CircleDot;
  return <Icon aria-hidden="true" className="h-4 w-4" />;
}
