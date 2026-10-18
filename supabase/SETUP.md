# Supabase setup (about 5 minutes)

1. Go to https://supabase.com, sign in, **New project** (any name, pick a database password and a nearby region). Wait until it finishes creating.
2. Open **SQL Editor > New query**, paste the contents of `supabase/schema.sql`, click **Run**. You should see "Success".
3. Open **Project Settings > API**. Copy the **Project URL** and the **service_role** key (the secret one, NOT the anon key).
4. In `D:\Alzheimers`, create a file named `.env` (it is git-ignored) with:

       SUPABASE_URL=https://xxxxxxxx.supabase.co
       SUPABASE_SERVICE_KEY=<service_role key>
       DOCTOR_USER=<pick a username>
       DOCTOR_PASSWORD=<pick a password>

5. Restart the app (`run_app.bat`). The yellow banner about Supabase should disappear.

Notes
- The service_role key gives full access. Keep it only in `.env`; never paste it in the browser, chat, or git. The browser never sees it.
- The page talks only to this local server; the server talks to Supabase. Tables have row-level security on with no public policies.
- Patient names and health answers are personal data. Use real patients only with their consent and your institution's approval.
- Without `.env`, the app saves to `data/local_store.json` instead.
