"use client";

import { useState } from "react";
import { useAuthUser } from "@/lib/auth";
import { supabase } from "@/lib/supabase";
import { LoginModal } from "./LoginModal";

export function AccountControl() {
  const [open, setOpen] = useState(false);
  const { user, signOut } = useAuthUser();

  if (!supabase) return null;

  if (user && !user.isAnonymous) {
    return (
      <div className="flex items-center gap-2 text-sm">
        <span className="font-semibold text-foreground/80">
          {user.name ?? user.email}
        </span>
        <button
          onClick={signOut}
          className="text-foreground/40 underline underline-offset-2"
        >
          Sign out
        </button>
      </div>
    );
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="rounded-full border-2 border-brand-blue px-3 py-1.5 text-sm font-bold text-brand-blue"
      >
        Save progress
      </button>
      {open && <LoginModal onClose={() => setOpen(false)} />}
    </>
  );
}
