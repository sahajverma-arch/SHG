import Link from "next/link";
import { RecentProgress } from "@/components/RecentProgress";
import { TodayStatus } from "@/components/TodayStatus";

const EXERCISES = [
  {
    href: "/reading-aloud",
    title: "पढ़कर सुनाओ",
    subtitle: "Reading aloud",
    blurb: "Read a passage out loud and see which words landed.",
  },
  {
    href: "/word-power",
    title: "शब्द शक्ति",
    subtitle: "Word power",
    blurb: "Match each Hindi word to what it means.",
  },
];

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-2xl flex-1 px-4 py-7 sm:px-6 sm:py-10">
      <h1 className="text-2xl font-semibold sm:text-3xl">नमस्ते!</h1>
      <p className="mt-1.5 text-[0.9375rem] text-muted">
        Two ways to practise today.
      </p>

      <div className="mt-5 space-y-3">
        <TodayStatus />
        <RecentProgress />
      </div>

      <ul className="mt-4 space-y-3">
        {EXERCISES.map((ex) => (
          <li key={ex.href}>
            <Link
              href={ex.href}
              className="card flex items-center gap-4 p-4 transition-colors hover:border-muted active:bg-border/25 sm:p-5"
            >
              <div className="min-w-0 flex-1">
                <p className="eyebrow">{ex.subtitle}</p>
                <p className="mt-1 font-[family-name:var(--font-read)] text-xl leading-snug sm:text-2xl">
                  {ex.title}
                </p>
                <p className="mt-1.5 text-sm leading-snug text-muted">
                  {ex.blurb}
                </p>
              </div>
              <span className="shrink-0 text-lg text-muted" aria-hidden>
                →
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
