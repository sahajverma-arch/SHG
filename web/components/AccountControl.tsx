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
          className="text-foreground/40 underline underline-offset-2 hover:text-foreground/60"
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
        className="btn btn-outline clr-blue text-sm"
      >
        Save progress
      </button>
      {open && <LoginModal onClose={() => setOpen(false)} />}
    </>
  );
}
