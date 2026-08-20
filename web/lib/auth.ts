"use client";

import { useEffect, useState } from "react";
import type { User } from "@supabase/supabase-js";
import { supabase } from "./supabase";
import { ensureAnonymousSession } from "./session";

export type AuthUser = {
  id: string;
  email: string | null;
  name: string | null;
  isAnonymous: boolean;
};

function toAuthUser(user: User | null | undefined): AuthUser | null {
  if (!user) return null;
  return {
    id: user.id,
    email: user.email ?? null,
    name: (user.user_metadata?.name as string | undefined) ?? null,
    isAnonymous: user.is_anonymous ?? false,
  };
}

export function useAuthUser() {
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    const client = supabase;
    if (!client) return;

    ensureAnonymousSession().then(() => {
      client.auth.getUser().then(({ data }) => setUser(toAuthUser(data.user)));
    });

    const { data: sub } = client.auth.onAuthStateChange((_event, session) => {
      setUser(toAuthUser(session?.user));
    });

    return () => sub.subscription.unsubscribe();
  }, []);

  async function signOut() {
    await supabase?.auth.signOut();
  }

  return { user, signOut };
}

/**
 * Step 1 of login: attaches an email (+ name) to the current session.
 *
 * If this browser is still anonymous, this upgrades that same user in
 * place — the streak already tied to it carries over, no migration
 * needed. If the email already belongs to a different, existing account,
 * updateUser rejects it — in that case this falls back to emailing that
 * existing account a sign-in code instead, so returning users land back
 * on their real streak rather than getting stuck.
 *
 * Requires "Anonymous sign-ins" AND "Allow manual linking" enabled in
 * Supabase (Authentication → Sign In / Providers).
 */
export async function startLogin(email: string, name: string): Promise<void> {
  if (!supabase) throw new Error("Supabase isn't configured yet.");

  await ensureAnonymousSession();

  const { error: linkError } = await supabase.auth.updateUser({
    email,
    data: { name },
  });

  if (linkError) {
    const { error: otpError } = await supabase.auth.signInWithOtp({ email });
    if (otpError) throw otpError;
  }
}

/** Step 2 of login: confirms the 6-digit code emailed in startLogin(). */
export async function confirmLogin(email: string, token: string): Promise<void> {
  if (!supabase) throw new Error("Supabase isn't configured yet.");

  const { error } = await supabase.auth.verifyOtp({
    email,
    token,
    type: "email",
  });
  if (error) throw error;
}
