"use client";

import Link from "next/link";
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
      <main className="mx-auto w-full max-w-2xl flex-1 px-4 py-6 sm:px-6 sm:py-10">
        <Celebrate message={`शाबाश! ${score} out of ${order.length}`} />
        <div className="mt-4 flex gap-3">
          <button onClick={playAgain} className="btn btn-primary flex-1">
            Play again
          </button>
          <Link href="/" className="btn btn-quiet flex-1">
            Done
          </Link>
        </div>
      </main>
    );
  }

  return (
    <main className="mx-auto w-full max-w-2xl flex-1 px-4 py-6 sm:px-6 sm:py-10">
      <p className="eyebrow">Word power · {index + 1} of {order.length}</p>
      <h1 className="mt-1 text-2xl font-semibold sm:text-3xl">शब्द शक्ति</h1>

      <div className="mt-4 flex h-1 gap-1">
        {order.map((_, i) => (
          <span
            key={i}
            className={`flex-1 rounded-full transition-colors duration-300 ${
              i < index ? "bg-accent" : i === index ? "bg-accent/45" : "bg-border"
            }`}
          />
        ))}
      </div>

      <div className="reading-surface mt-5 px-4 py-12 text-center">
        <p className="font-[family-name:var(--font-read)] text-4xl leading-tight sm:text-5xl">
          {question.word}
        </p>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-2.5">
        {shuffledOptions.map((option) => {
          const isCorrect = option === question.correct;
          const isPicked = option === selected;
          const revealed = selected !== null;

          const stateClass = !revealed
            ? "btn-quiet"
            : isCorrect
              ? "border-good bg-good/10 text-good"
              : isPicked
                ? "border-warn bg-warn/10 text-warn"
                : "btn-quiet opacity-40";

          return (
            <button
              key={option}
              onClick={() => choose(option)}
              disabled={revealed}
              className={`btn w-full border ${stateClass}`}
            >
              {option}
            </button>
          );
        })}
      </div>
    </main>
  );
}
