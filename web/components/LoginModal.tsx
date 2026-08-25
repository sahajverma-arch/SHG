"use client";

import { useState, type FormEvent } from "react";
import { confirmLogin, startLogin } from "@/lib/auth";

type Step = "form" | "otp" | "success";

export function LoginModal({ onClose }: { onClose: () => void }) {
  const [step, setStep] = useState<Step>("form");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

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

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-foreground/30 backdrop-blur-sm sm:items-center sm:px-6">
      <div className="card relative w-full rounded-b-none p-5 pb-[max(1.25rem,env(safe-area-inset-bottom))] sm:max-w-sm sm:rounded-b-[0.875rem] sm:pb-5">
        <button
          onClick={onClose}
          aria-label="Close"
          className="absolute right-3 top-3 flex h-9 w-9 items-center justify-center rounded-lg text-muted hover:text-foreground"
        >
          ✕
        </button>

        {step === "form" && (
          <form onSubmit={submitForm} className="flex flex-col gap-3">
            <h2 className="text-lg font-semibold">Save your streak</h2>
            <p className="text-sm text-muted">
              So it&apos;s still here next time, on any device.
            </p>
            <input
              required
              placeholder="Your name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="min-h-[2.875rem] rounded-lg border border-border bg-surface px-3 py-2 text-base outline-none transition-colors focus:border-accent"
            />
            <input
              required
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="min-h-[2.875rem] rounded-lg border border-border bg-surface px-3 py-2 text-base outline-none transition-colors focus:border-accent"
            />
            {error && <p className="text-sm text-warn">{error}</p>}
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
            <h2 className="text-lg font-semibold">Check your email</h2>
            <p className="text-sm text-muted">
              Enter the 6-digit code sent to {email}.
            </p>
            <input
              required
              inputMode="numeric"
              placeholder="123456"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              className="min-h-[2.875rem] rounded-lg border border-border bg-surface px-3 py-2 text-center text-lg tracking-widest outline-none transition-colors focus:border-accent"
            />
            {error && <p className="text-sm text-warn">{error}</p>}
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
            <p className="text-base font-semibold text-good">You&apos;re all set!</p>
          </div>
        )}
      </div>
    </div>
  );
}
