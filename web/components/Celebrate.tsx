export function Celebrate({ message }: { message: string }) {
  return (
    <div className="animate-[pop_0.4s_ease-out] rounded-3xl border-2 border-brand-green bg-brand-green/10 px-6 py-8 text-center text-xl font-bold text-brand-green">
      {message}
    </div>
  );
}
