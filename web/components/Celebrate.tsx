export function Celebrate({ message }: { message: string }) {
  return (
    <div className="card fade-up p-6 text-center">
      <p className="text-lg font-semibold text-good">{message}</p>
    </div>
  );
}
