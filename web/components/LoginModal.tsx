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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/40 px-6">
      <div className="relative w-full max-w-sm rounded-3xl border-2 border-brand-blue bg-surface p-6">
        <button
          onClick={onClose}
          aria-label="Close"
          className="absolute right-4 top-4 text-foreground/40"
        >
          ✕
        </button>

        {step === "form" && (
          <form onSubmit={submitForm} className="flex flex-col gap-3">
            <h2 className="font-[family-name:var(--font-display)] text-xl text-brand-blue">
              Save your streak
            </h2>
            <p className="text-sm text-foreground/60">
              So it&apos;s still here next time, on any device.
            </p>
            <input
              required
              placeholder="Your name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="rounded-xl border-2 border-border bg-background px-3 py-2 text-foreground"
            />
            <input
              required
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="rounded-xl border-2 border-border bg-background px-3 py-2 text-foreground"
            />
            {error && <p className="text-sm font-semibold text-brand-pink">{error}</p>}
            <button
              disabled={busy}
              className="mt-1 rounded-full bg-brand-blue px-4 py-2 font-bold text-white disabled:opacity-50"
            >
              {busy ? "Sending…" : "Send code"}
            </button>
          </form>
        )}

        {step === "otp" && (
          <form onSubmit={submitOtp} className="flex flex-col gap-3">
            <h2 className="font-[family-name:var(--font-display)] text-xl text-brand-blue">
              Check your email
            </h2>
            <p className="text-sm text-foreground/60">
              Enter the 6-digit code sent to {email}.
            </p>
            <input
              required
              inputMode="numeric"
              placeholder="123456"
              value={otp}
              onChange={(e) => setOtp(e.target.value)}
              className="rounded-xl border-2 border-border bg-background px-3 py-2 text-center text-lg tracking-widest text-foreground"
            />
            {error && <p className="text-sm font-semibold text-brand-pink">{error}</p>}
            <button
              disabled={busy}
              className="mt-1 rounded-full bg-brand-blue px-4 py-2 font-bold text-white disabled:opacity-50"
            >
              {busy ? "Checking…" : "Confirm"}
            </button>
          </form>
        )}

        {step === "success" && (
          <p className="py-6 text-center text-lg font-bold text-brand-green">
            You&apos;re all set! 🎉
          </p>
        )}
      </div>
    </div>
  );
}
