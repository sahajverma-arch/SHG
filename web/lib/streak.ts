"use client";

import { useCallback, useEffect, useState } from "react";
import { supabase } from "./supabase";
import { ensureAnonymousSession } from "./session";

const LOCAL_KEY = "shj_streak";

type StreakRow = { count: number; last_practiced_on: string | null };

function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function yesterdayKey(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10);
}

function advance(row: StreakRow): StreakRow {
  const today = todayKey();
  if (row.last_practiced_on === today) return row;
  const count = row.last_practiced_on === yesterdayKey() ? row.count + 1 : 1;
  return { count, last_practiced_on: today };
}

function displayCount(row: StreakRow): number {
  if (!row.last_practiced_on) return 0;
  const stillLive =
    row.last_practiced_on === todayKey() || row.last_practiced_on === yesterdayKey();
  return stillLive ? row.count : 0;
}

// ---- localStorage fallback, used only when no Supabase project is configured ----

function loadLocal(): StreakRow {
  if (typeof window === "undefined") return { count: 0, last_practiced_on: null };
  try {
    const raw = window.localStorage.getItem(LOCAL_KEY);
    return raw ? (JSON.parse(raw) as StreakRow) : { count: 0, last_practiced_on: null };
  } catch {
    return { count: 0, last_practiced_on: null };
  }
}

function saveLocal(row: StreakRow) {
  window.localStorage.setItem(LOCAL_KEY, JSON.stringify(row));
}

// ---- Supabase-backed persistence, via an anonymous auth session ----
// (the same session that lib/auth.ts upgrades to a permanent one on login)

async function loadRemote(userId: string): Promise<StreakRow> {
  const { data } = await supabase!
    .from("streaks")
    .select("count, last_practiced_on")
    .eq("user_id", userId)
    .maybeSingle();

  return data ?? { count: 0, last_practiced_on: null };
}

async function saveRemote(userId: string, row: StreakRow) {
  await supabase!.from("streaks").upsert({
    user_id: userId,
    count: row.count,
    last_practiced_on: row.last_practiced_on,
    updated_at: new Date().toISOString(),
  });
}

export function useStreak() {
  const [row, setRow] = useState<StreakRow>({ count: 0, last_practiced_on: null });
  const [userId, setUserId] = useState<string | null>(null);
  const [practicedToday, setPracticedToday] = useState(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (!supabase) {
        const local = loadLocal();
        if (!cancelled) {
          setRow(local);
          setPracticedToday(local.last_practiced_on === todayKey());
        }
        return;
      }

      const id = await ensureAnonymousSession();
      if (!id) return;
      const remote = await loadRemote(id);
      if (!cancelled) {
        setUserId(id);
        setRow(remote);
        setPracticedToday(remote.last_practiced_on === todayKey());
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const complete = useCallback(async () => {
    const next = advance(row);
    setRow(next);
    setPracticedToday(true);

    if (!supabase) {
      saveLocal(next);
      return;
    }
    if (userId) {
      await saveRemote(userId, next);
    }
  }, [row, userId]);

  return { streak: displayCount(row), practicedToday, complete };
}
