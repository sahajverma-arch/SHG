export function Celebrate({ message }: { message: string }) {
  return (
    <div className="card relative animate-[pop_0.4s_ease-out] overflow-hidden border-brand-green/30 bg-gradient-to-br from-brand-green/10 via-surface to-brand-yellow/10 px-6 py-8 text-center">
      <span className="icon-badge float mx-auto h-14 w-14 bg-brand-green/15 text-3xl">
        🎉
      </span>
      <p className="mt-3 text-xl font-bold text-brand-green">{message}</p>
    </div>
  );
}
