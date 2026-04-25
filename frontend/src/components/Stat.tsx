export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-24">
      <div className="text-xs uppercase tracking-wide text-stone-500">{label}</div>
      <div className="text-lg font-semibold text-stone-950">{value}</div>
    </div>
  );
}
