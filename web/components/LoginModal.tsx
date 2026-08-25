"use client";

import { useEffect, useState, type FormEvent } from "react";
import { createPortal } from "react-dom";
import { confirmLogin, startLogin } from "@/lib/auth";

type Step = "form" | "otp" | "success";

export function LoginModal({ onClose }: { onClose: () => void }) {
  const [step, setStep] = useState<Step>("form");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  // Escape closes it. Every other dialog on the web does, so its absence reads
  // as the page having frozen rather than as a deliberate choice.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function submitForm(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await startLogin(email, name);
      setStep("otp");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  async function submitOtp(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await confirmLogin(email, otp);
      setStep("success");
      setTimeout(onClose, 1200);
    } catch (err) {
      setError(err instanceof Error ? err.message : "That code didn't work");
    } finally {
      setBusy(false);
    }
  }

  if (!mounted) return null;

  // Rendered into <body>, not where it sits in the tree. This component is
  // mounted from AccountControl, inside the sticky header — and that header
  // carries `backdrop-blur`. A backdrop-filter makes an element a containing
  // block for `position: fixed` descendants, so `fixed inset-0` was resolving
  // against the 66px header box rather than the viewport: the overlay was
  // 66px tall and the dialog, centred in it, hung off the top of the screen
  // with its heading and close button out of reach at every window size.
  return createPortal(
    // Closing is keyed to where the press *started*, not where it was released:
    // dragging to select text inside a field and letting go over the dim area
    // would otherwise throw the half-filled form away.
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-foreground/30 backdrop-blur-sm sm:items-center sm:px-6"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {/* max-h + overflow because a short viewport — a phone in landscape — cut
          the heading and the close button off the top with no way to reach
          them: the overlay is fixed and nothing inside it could scroll.
          overscroll-contain stops that scroll running on into the page behind
          once it bottoms out. */}
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="login-modal-title"
        className="card relative max-h-[85dvh] w-full overflow-y-auto overscroll-contain rounded-b-none p-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] sm:max-w-sm sm:rounded-b-[0.875rem] sm:pb-5"
      >
        <button
          onClick={onClose}
          aria-label="Close"
          className="absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-lg text-muted hover:text-foreground"
        >
          ✕
        </button>

        {step === "form" && (
          <form onSubmit={submitForm} className="flex flex-col gap-3">
            <h2 id="login-modal-title" className="text-lg font-semibold">Save your streak</h2>
            <p className="text-sm text-muted">
              So it&apos;s still here next time, on any device.
            </p>
            {/* Labelled for assistive tech but not shown: a placeholder is not
                a label — it disappears the moment a character is typed. The
                `outline-none` these carried has gone too, so the global
                :focus-visible ring applies; a border tint alone was not enough
                to see where focus had landed. */}
            <input
              required
              aria-label="Your name"
              placeholder="Your name"
              autoComplete="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="min-h-[2.875rem] rounded-lg border border-border bg-surface px-3 py-2 text-base transition-colors focus:border-accent"
            />
            <input
              required
              type="email"
              aria-label="Email"
              placeholder="Email"
              autoComplete="email"
              spellCheck={false}
              autoCapitalize="none"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="min-h-[2.875rem] rounded-lg border border-border bg-surface px-3 py-2 text-base transition-colors focus:border-accent"
            />
            <p className="text-sm text-warn empty:hidden" aria-live="polite">
              {error}
            </p>
            <button
              disabled={busy}
              className="btn btn-primary mt-1 w-full"
            >
              {busy ? "Sending…" : "Send code"}
            </button>
          </form>
        )}

        {step === "otp" && (
          <form onSubmit={submitOtp} className="flex flex-col gap-3">
            <h2 id="login-modal-title" className="text-lg font-semibold">Check your email</h2>
            <p className="text-sm text-muted">
              Enter the 6-digit code sent to {email}.
            </p>
            {/* `one-time-code` is what lets a phone offer the code straight
                from the notification instead of making a parent switch apps to
                copy it. spellCheck off keeps the keyboard from correcting
                digits. */}
            <input
              required
              inputMode="numeric"
              aria-label="6-digit code"
              placeholder="123456"
              autoComplete="one-time-code"
              spellCheck={false}
              maxLength={6}
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              className="min-h-[2.875rem] rounded-lg border border-border bg-surface px-3 py-2 text-center text-lg tracking-widest tabular-nums transition-colors focus:border-accent"
            />
            <p className="text-sm text-warn empty:hidden" aria-live="polite">
              {error}
            </p>
            <button
              disabled={busy}
              className="btn btn-primary mt-1 w-full"
            >
              {busy ? "Checking…" : "Confirm"}
            </button>
          </form>
        )}

        {step === "success" && (
          <div className="py-6 text-center">
            <p id="login-modal-title" className="text-base font-semibold text-good">You&apos;re all set!</p>
          </div>
        )}
      </div>
    </div>,
    document.body
  );
}
