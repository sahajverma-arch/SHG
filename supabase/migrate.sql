-- Minimal migration: brings an existing project up to the 20-point rubric.
--
-- Use this instead of re-running schema.sql when the tables already exist.
-- It only adds columns and relaxes constraints -- no create table, no
-- policies -- so it cannot collide with anything already in the database.
--
-- Safe to re-run.

alter table passages add column if not exists level text not null default 'P1-P2';

alter table attempts add column if not exists total_score smallint;
alter table attempts add column if not exists pronunciation_score smallint;
alter table attempts add column if not exists pace_score smallint;
alter table attempts add column if not exists pre_coverage_total smallint;
alter table attempts add column if not exists coverage_percent smallint;
alter table attempts add column if not exists transcript text;
alter table attempts add column if not exists word_analysis jsonb;

-- Nothing uploads recordings to Storage yet, so a NOT NULL here blocks
-- writing an attempt at all.
alter table attempts alter column audio_url drop not null;

-- Legacy columns from before the rubric. Guarded, so this is also a no-op on
-- a project that never had them.
do $$
begin
  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'attempts'
      and column_name = 'accuracy_score'
  ) then
    alter table attempts alter column accuracy_score drop not null;
  end if;

  if exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'attempts'
      and column_name = 'fluency_score' and data_type = 'numeric'
  ) then
    alter table attempts alter column fluency_score drop not null;
    alter table attempts alter column fluency_score type smallint
      using round(fluency_score);
  end if;
end $$;
