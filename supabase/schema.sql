-- Run once in Supabase: SQL Editor > New query > paste > Run.
create table if not exists patients (
  code text primary key,
  name text not null,
  questionnaire jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create table if not exists screenings (
  id bigint generated always as identity primary key,
  code text not null references patients(code),
  results jsonb,
  combined jsonb,
  summary jsonb,
  clock_jpeg text,
  created_at timestamptz not null default now()
);
-- Row-level security ON with no policies: the public (anon) key can read nothing.
-- Only the server, using the service_role key from .env, can read or write.
alter table patients enable row level security;
alter table screenings enable row level security;
