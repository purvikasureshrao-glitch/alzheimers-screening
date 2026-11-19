# 3. Database — where records are kept

The app can save records in two ways. It chooses automatically when it starts (`app/store.py`).

| Mode | When | Where |
|---|---|---|
| **Supabase** (cloud) | A `.env` file with Supabase keys exists | A PostgreSQL database online |
| **Local file** | No `.env` | `data/local_store.json` on this laptop |

The yellow banner at the top of the page tells you which one is active. The app works fully either way.

## What is Supabase?

Supabase is a free hosted database service (PostgreSQL). You create a free project, run a small SQL file once,
and the app can store and read records there. Records then live online and can be looked up from more than one
computer (everyone using the same keys).

## The three tables (`supabase/schema.sql`)

| Table | Columns |
|---|---|
| `patients` | `code` (unique ID), `name`, `questionnaire` (all 15 answers), `created_at` |
| `screenings` | `id`, `code` (which patient), `results` (the four test results), `combined` (how many tests were flagged), `summary` (written summary), `clock_jpeg` (the drawing), `created_at` |
| `doctors` | `username`, `full_name`, `salt`, `pw_hash`, `recovery_salt`, `recovery_hash`, `created_at` |

## Security, in plain words

- The server talks to Supabase with a **service-role key**. It lives only in a file called `.env` on the laptop. The browser never receives it.
- `.env` is **never** uploaded to GitHub and **never** included in the shared zip. Everyone makes their own.
- **Row-level security** is switched on with no public rules, so someone holding only the public key can read nothing.
- **Doctor passwords are never stored.** Only a salted `scrypt` hash is stored (a one-way scramble). The same goes for the recovery code.
- At sign-up the doctor receives a one-time **recovery code** (like `AB3D-EF7G-HJ2K`). With it they can reset a forgotten password without e-mail. A new code is issued after each reset.
- Patient names and health answers are personal data. Use real patients only with consent and institutional approval.

## Setting up Supabase (about 5 minutes, optional)

1. Go to supabase.com, sign in, click **New project**.
2. **SQL Editor → New query**, paste everything from `supabase/schema.sql`, click **Run**.
3. **Project Settings → API**: copy the **Project URL** and the **service_role** key (the secret one, not "anon").
4. In the project folder copy `.env.example` to a new file named exactly `.env` and fill in the two values.
5. Restart the app. The banner should now say Supabase.

## Why is `.env.example` "blank"?

`.env.example` is only a **template** with placeholder text such as `paste-your-service-role-key-here`. It has to
contain no real secrets, because it is the file that gets shared. Your real keys go in `.env`, which stays on your
own computer only.
