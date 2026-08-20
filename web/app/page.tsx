import Link from "next/link";

const EXERCISES = [
  {
    href: "/reading-aloud",
    emoji: "🎙️",
    title: "Reading Aloud",
    subtitle: "Read a line out loud, get scored",
    cardClassName: "border-brand-pink bg-brand-pink/10 hover:bg-brand-pink/20",
    titleClassName: "text-brand-pink",
  },
  {
    href: "/word-power",
    emoji: "🧩",
    title: "शब्द शक्ति",
    subtitle: "Match each word to its meaning",
    cardClassName: "border-brand-blue bg-brand-blue/10 hover:bg-brand-blue/20",
    titleClassName: "text-brand-blue",
  },
];

export default function Home() {
  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 px-6 py-12">
      <div>
        <h1 className="font-[family-name:var(--font-display)] text-3xl text-brand-pink">
          नमस्ते! 👋
        </h1>
        <p className="mt-1 text-foreground/60">
          Pick an exercise to keep your streak going.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {EXERCISES.map((ex) => (
          <Link
            key={ex.href}
            href={ex.href}
            className={`flex flex-col gap-2 rounded-3xl border-2 p-6 transition-colors ${ex.cardClassName}`}
          >
            <span className="text-4xl">{ex.emoji}</span>
            <span
              className={`font-[family-name:var(--font-display)] text-xl ${ex.titleClassName}`}
            >
              {ex.title}
            </span>
            <span className="text-sm text-foreground/60">{ex.subtitle}</span>
          </Link>
        ))}
      </div>
    </main>
  );
}
