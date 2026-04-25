import type { ButtonHTMLAttributes, AnchorHTMLAttributes, ReactNode } from "react";

import { cn } from "../../lib/utils";

const buttonClass =
  "inline-flex h-10 items-center justify-center gap-2 rounded-md border border-stone-300 bg-white px-3 text-sm font-medium text-stone-900 shadow-sm transition hover:bg-stone-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-600 disabled:pointer-events-none disabled:opacity-50";

export function Button({
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={cn(buttonClass, className)} {...props} />;
}

export function ButtonLink({
  className,
  children,
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement> & { children: ReactNode }) {
  return (
    <a className={cn(buttonClass, className)} {...props}>
      {children}
    </a>
  );
}
