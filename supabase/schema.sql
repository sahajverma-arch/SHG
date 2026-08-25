-- Prototype schema — matches blueprint §05.
-- Run in the Supabase SQL editor for a new project.

create extension if not exists "uuid-ossp";

create table if not exists passages (
  id uuid primary key default uuid_generate_v4(),
  text_hi text not null,
  audio_model_url text,
  difficulty smallint not null default 1 check (difficulty between 1 and 5),
  -- Level band drives the expected reading pace used by the pace score.
  level text not null default 'P1-P2',
  created_at timestamptz not null default now()
);

create table if not exists vocab_words (
  id uuid primary key default uuid_generate_v4(),
  word_hi text not null,
  meaning text not null,
  audio_url text,
  created_at timestamptz not null default now()
);

-- Scores are the 20-point rubric: pronunciation (8) + fluency (6) + pace (6),
-- scaled by how much of the passage was attempted. The pre_coverage_* columns
-- keep the unscaled figures so progress can be tracked separately from how
-- much a learner got through.
--
-- audio_url is nullable: nothing uploads recordings to Storage yet, and a NOT
-- NULL here is what would block writing an attempt at all.
create table if not exists attempts (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  passage_id uuid not null references passages(id) on delete cascade,
  audio_url text,
  total_score smallint,
  pronunciation_score smallint,
  fluency_score smallint,
  pace_score smallint,
  pre_coverage_total smallint,
  coverage_percent smallint,
  words_per_minute numeric(6,1),
  transcript text,
  word_analysis jsonb,
  created_at timestamptz not null default now()
);

create table if not exists dictionary_entries (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  word_id uuid not null references vocab_words(id) on delete cascade,
  source_module text not null,
  saved_at timestamptz not null default now(),
  unique (user_id, word_id)
);

create table if not exists streaks (
  user_id uuid primary key references auth.users(id) on delete cascade,
  count integer not null default 0,
  last_practiced_on date,
  updated_at timestamptz not null default now()
);

create table if not exists subscriptions (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null unique references auth.users(id) on delete cascade,
  plan text not null default 'free',
  status text not null default 'active',
  billing_cycle text,
  updated_at timestamptz not null default now()
);

-- Row Level Security: everyone only ever sees their own attempts/library/plan.
alter table attempts enable row level security;
alter table dictionary_entries enable row level security;
alter table streaks enable row level security;
alter table subscriptions enable row level security;

-- Postgres has no `create policy if not exists`, so each one is dropped first.
-- Without this the whole script aborts on a re-run and never reaches the
-- migration block at the bottom.
drop policy if exists "attempts_owner_rw" on attempts;
create policy "attempts_owner_rw" on attempts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "dictionary_owner_rw" on dictionary_entries;
create policy "dictionary_owner_rw" on dictionary_entries
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "streaks_owner_rw" on streaks;
create policy "streaks_owner_rw" on streaks
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "subscriptions_owner_rw" on subscriptions;
create policy "subscriptions_owner_rw" on subscriptions
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- passages and vocab_words are shared reference content: readable by any
-- signed-in user, writable only via the Supabase dashboard/service role.
alter table passages enable row level security;
alter table vocab_words enable row level security;

drop policy if exists "passages_read" on passages;
create policy "passages_read" on passages for select using (auth.role() = 'authenticated');

drop policy if exists "vocab_words_read" on vocab_words;
create policy "vocab_words_read" on vocab_words for select using (auth.role() = 'authenticated');

-- ---------------------------------------------------------------------------
-- Migration for projects created before the 20-point rubric.
-- Safe to re-run; `create table if not exists` above will not alter an
-- existing table, so the changes have to be spelled out here too.
-- ---------------------------------------------------------------------------

alter table passages add column if not exists level text not null default 'P1-P2';

alter table attempts add column if not exists total_score smallint;
alter table attempts add column if not exists pronunciation_score smallint;
alter table attempts add column if not exists pace_score smallint;
alter table attempts add column if not exists pre_coverage_total smallint;
alter table attempts add column if not exists coverage_percent smallint;
alter table attempts add column if not exists transcript text;
alter table attempts add column if not exists word_analysis jsonb;

-- Nothing uploads to Storage yet, so a NOT NULL here blocks writing an attempt
-- at all. Safe on a fresh project too: dropping an absent NOT NULL is a no-op.
alter table attempts alter column audio_url drop not null;

-- The rest only exist on projects created before the rubric, so they are
-- guarded — on a fresh project these columns are already correct or absent,
-- and running them unconditionally would abort the script.
do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'attempts'
      and column_name = 'accuracy_score'
  ) then
    -- Replaced by the pronunciation score plus coverage_percent.
    alter table attempts alter column accuracy_score drop not null;
  end if;

  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'attempts'
      and column_name = 'fluency_score' and data_type = 'numeric'
  ) then
    -- Was numeric(5,1) as a percentage; now a 0-6 integer out of the rubric.
    alter table attempts alter column fluency_score drop not null;
    alter table attempts alter column fluency_score type smallint
      using round(fluency_score);
  end if;
end $$;
