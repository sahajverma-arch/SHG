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
      <main className="mx-auto max-w-2xl px-6 py-14">
        <Celebrate message={`शाबाश! You scored ${score} / ${order.length}`} />
        <button
          onClick={playAgain}
          className="btn btn-solid clr-blue mx-auto mt-6 block text-sm"
        >
          Play again
        </button>
      </main>
    );
  }

  return (
    <main className="mx-auto max-w-2xl px-6 py-14">
      <div className="flex items-center gap-3">
        <span className="icon-badge h-11 w-11 bg-brand-blue/15 text-xl">
          🧩
        </span>
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-2xl text-brand-blue">
            शब्द शक्ति
          </h1>
          <p className="text-sm text-foreground/60">
            Question {index + 1} of {order.length}
          </p>
        </div>
      </div>

      <div className="mt-2 flex gap-1.5">
        {order.map((_, i) => (
          <span
            key={i}
            className={`h-1.5 flex-1 rounded-full transition-colors ${
              i < index
                ? "bg-brand-blue"
                : i === index
                  ? "bg-brand-blue/50"
                  : "bg-border"
            }`}
          />
        ))}
      </div>

      <div className="card mt-6 p-8 text-center">
        <p className="text-4xl font-bold text-foreground">{question.word}</p>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3">
        {shuffledOptions.map((option) => {
          const isCorrect = option === question.correct;
          const isPicked = option === selected;
          const revealed = selected !== null;

          const stateClass = !revealed
            ? "btn-outline clr-blue"
            : isCorrect
              ? "btn-solid clr-green"
              : isPicked
                ? "btn-solid clr-pink"
                : "btn-outline clr-blue opacity-40";

          return (
            <button
              key={option}
              onClick={() => choose(option)}
              disabled={revealed}
              className={`btn ${stateClass} w-full text-base`}
            >
              {option}
            </button>
          );
        })}
      </div>
    </main>
  );
}
