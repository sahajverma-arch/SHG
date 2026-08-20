"use client";

import { supabase } from "./supabase";

/**
 * Every visitor gets a Supabase anonymous session on first load — this is
 * what lets the streak persist without a login screen. Logging in with
 * email later upgrades this same session in place (see lib/auth.ts),
 * rather than starting a new user id.
 *
 * Multiple components (StreakBadge, AccountControl) call this on mount.
 * Without de-duping, two concurrent callers can each race into their own
 * signInAnonymously(), leaving one component holding a user id that no
 * longer matches the session the shared Supabase client actually ends up
 * with — every write from that component then fails RLS, since the
 * request is authenticated as a different user than the row it's writing.
 * The module-level promise makes every caller within a page load await
 * the exact same sign-in attempt.
 */
let sessionPromise: Promise<string | null> | null = null;

export function ensureAnonymousSession(): Promise<string | null> {
  if (!supabase) return Promise.resolve(null);
  const client = supabase;

  if (!sessionPromise) {
    sessionPromise = (async () => {
      const { data } = await client.auth.getSession();
      if (data.session) return data.session.user.id;

      const { data: signInData, error } = await client.auth.signInAnonymously();
      if (error) {
        console.error(
          "Anonymous sign-in failed — enable it in Supabase dashboard → Authentication → Sign In / Providers → Anonymous.",
          error
        );
        sessionPromise = null; // allow a retry on the next call
        return null;
      }
      return signInData.user?.id ?? null;
    })();
  }

  return sessionPromise;
}
