import Link from "next/link";

const EXERCISES = [
  {
    href: "/reading-aloud",
    emoji: "🎙️",
    title: "Reading Aloud",
    subtitle: "Read a line out loud, get scored",
    badgeClassName: "bg-brand-pink/15",
    titleClassName: "text-brand-pink",
  },
  {
    href: "/word-power",
    emoji: "🧩",
    title: "शब्द शक्ति",
    subtitle: "Match each word to its meaning",
    badgeClassName: "bg-brand-blue/15",
    titleClassName: "text-brand-blue",
  },
];

export default function Home() {
  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-10 px-6 py-14">
      <div className="flex flex-col items-start gap-3">
        <span className="icon-badge float h-14 w-14 bg-brand-yellow/25 text-3xl">
          👋
        </span>
        <h1 className="font-[family-name:var(--font-display)] text-4xl text-brand-pink">
          नमस्ते!
        </h1>
        <p className="text-base text-foreground/60">
          Pick an exercise to keep your streak going.
        </p>
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        {EXERCISES.map((ex) => (
          <Link
            key={ex.href}
            href={ex.href}
            className="card group flex flex-col gap-4 p-6 transition-all duration-200 hover:-translate-y-1 hover:shadow-xl"
          >
            <span
              className={`icon-badge h-14 w-14 text-3xl ${ex.badgeClassName}`}
            >
              {ex.emoji}
            </span>
            <div>
              <p
                className={`font-[family-name:var(--font-display)] text-xl ${ex.titleClassName}`}
              >
                {ex.title}
              </p>
              <p className="mt-1 text-sm text-foreground/60">{ex.subtitle}</p>
            </div>
            <span
              className={`mt-auto inline-flex items-center gap-1 text-sm font-bold ${ex.titleClassName}`}
            >
              Start
              <span className="transition-transform duration-200 group-hover:translate-x-1">
                →
              </span>
            </span>
          </Link>
        ))}
      </div>
    </main>
  );
}
