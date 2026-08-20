"use client";

import { useEffect, useState } from "react";
import { Celebrate } from "@/components/Celebrate";
import { useStreak } from "@/lib/streak";

type Question = { word: string; correct: string; options: string[] };

const QUESTIONS: Question[] = [
  { word: "सूरज", correct: "Sun", options: ["Sun", "Moon", "Star", "Cloud"] },
  { word: "चाँद", correct: "Moon", options: ["Moon", "River", "Tree", "Fish"] },
  { word: "किताब", correct: "Book", options: ["Book", "Chair", "Table", "Door"] },
  { word: "पानी", correct: "Water", options: ["Water", "Fire", "Air", "Earth"] },
  { word: "फूल", correct: "Flower", options: ["Flower", "Leaf", "Root", "Seed"] },
  { word: "हाथी", correct: "Elephant", options: ["Elephant", "Lion", "Tiger", "Deer"] },
];

function shuffle<T>(items: T[]): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export default function WordPowerPage() {
  const { complete } = useStreak();
  // Stable QUESTIONS order on first render (server and client must match);
  // shuffled only after mount, so hydration never disagrees with itself.
  const [order, setOrder] = useState(QUESTIONS);
  useEffect(() => {
    setOrder(shuffle(QUESTIONS));
  }, []);
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [finished, setFinished] = useState(false);

  const question = order[index];
  // Same hydration-safety reasoning as `order` above.
  const [shuffledOptions, setShuffledOptions] = useState(question.options);
  useEffect(() => {
    setShuffledOptions(shuffle(question.options));
  }, [question]);

  function choose(option: string) {
    if (selected) return;
    setSelected(option);
    if (option === question.correct) setScore((s) => s + 1);

    setTimeout(() => {
      if (index + 1 < order.length) {
        setIndex((i) => i + 1);
        setSelected(null);
      } else {
        setFinished(true);
        complete();
      }
    }, 700);
  }

  function playAgain() {
    setIndex(0);
    setScore(0);
    setSelected(null);
    setFinished(false);
  }

  if (finished) {
    return (
      <main className="mx-auto max-w-2xl px-6 py-16">
        <Celebrate message={`शाबाश! You scored ${score} / ${order.length} 🎉`} />
        <button
          onClick={playAgain}
          className="mx-auto mt-6 block rounded-full bg-brand-blue px-6 py-2 text-sm font-bold text-white"
        >
          Play again
        </button>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <h1 className="font-[family-name:var(--font-display)] text-2xl text-brand-blue">
        शब्द शक्ति
      </h1>
      <p className="mt-2 text-sm text-foreground/60">
        Question {index + 1} of {order.length}
      </p>

      <p className="mt-8 rounded-3xl border-2 border-brand-blue bg-brand-blue/10 p-8 text-center text-4xl font-bold text-foreground">
        {question.word}
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3">
        {shuffledOptions.map((option) => {
          const isCorrect = option === question.correct;
          const isPicked = option === selected;
          const revealed = selected !== null;

          const stateClass = !revealed
            ? "border-border bg-surface hover:border-brand-blue"
            : isCorrect
              ? "border-brand-green bg-brand-green/15 text-brand-green"
              : isPicked
                ? "border-brand-pink bg-brand-pink/15 text-brand-pink"
                : "border-border bg-surface opacity-50";

          return (
            <button
              key={option}
              onClick={() => choose(option)}
              disabled={revealed}
              className={`rounded-2xl border-2 px-4 py-4 text-base font-semibold transition-colors ${stateClass}`}
            >
              {option}
            </button>
          );
        })}
      </div>
    </main>
  );
}
