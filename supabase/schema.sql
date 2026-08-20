-- Prototype schema — matches blueprint §05.
-- Run in the Supabase SQL editor for a new project.

create extension if not exists "uuid-ossp";

create table if not exists passages (
  id uuid primary key default uuid_generate_v4(),
  text_hi text not null,
  audio_model_url text,
  difficulty smallint not null default 1 check (difficulty between 1 and 5),
  created_at timestamptz not null default now()
);

create table if not exists vocab_words (
  id uuid primary key default uuid_generate_v4(),
  word_hi text not null,
  meaning text not null,
  audio_url text,
  created_at timestamptz not null default now()
);

create table if not exists attempts (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid not null references auth.users(id) on delete cascade,
  passage_id uuid not null references passages(id) on delete cascade,
  audio_url text not null,
  accuracy_score numeric(5,1) not null,
  fluency_score numeric(5,1) not null,
  words_per_minute numeric(6,1),
  word_diff jsonb,
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

create policy "attempts_owner_rw" on attempts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "dictionary_owner_rw" on dictionary_entries
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "streaks_owner_rw" on streaks
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "subscriptions_owner_rw" on subscriptions
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- passages and vocab_words are shared reference content: readable by any
-- signed-in user, writable only via the Supabase dashboard/service role.
alter table passages enable row level security;
alter table vocab_words enable row level security;

create policy "passages_read" on passages for select using (auth.role() = 'authenticated');
create policy "vocab_words_read" on vocab_words for select using (auth.role() = 'authenticated');
