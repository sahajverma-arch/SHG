import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

// Prototype runs fine without a Supabase project connected yet — pages fall
// back to placeholder content until these env vars are set.
export const supabase = url && anonKey ? createClient(url, anonKey) : null;
