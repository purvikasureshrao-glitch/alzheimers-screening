# How to use the project from GitHub (for the person who received the link)

You were sent a **GitHub link** to this project. This guide gets the app running on your laptop.
No programming knowledge is needed.

> This is a research / student screening aid. It does **not** diagnose anything.

## What you need

- **Windows 10 or 11**
- About **10 GB** free disk space
- **Internet** for the first setup (15–40 minutes)
- A **webcam** and **microphone**
- **Chrome** or **Edge**

## Important: GitHub does not hold everything

GitHub keeps the **code** only. The trained **models** are too big, so they are a separate download (Step 2).
Secret keys and private data are never on GitHub.

## Step 1 — Download the code

**Easiest way (no Git needed):**
1. Open the link you were sent.
2. Click the green **Code** button → **Download ZIP**.
3. Right-click the downloaded zip → **Extract All…** → choose a simple folder, e.g. `C:\Alzheimers`.

**Alternative (with Git):** install Git from git-scm.com, open **PowerShell**, and type:
```powershell
cd C:\
git clone https://github.com/OWNER/REPOSITORY.git Alzheimers
```
(Replace the address with the link you were sent.)

## Step 2 — Download the models

1. On the GitHub page click **Releases** (on the right side).
2. Under the newest release, under **Assets**, click **models.zip** to download it (about 400 MB).
3. Put `models.zip` **inside the project folder** (next to `setup_new_pc.bat`). **Do not unzip it** — the setup does that.

## Step 3 — One-time setup

1. Double-click **`setup_new_pc.bat`**.
2. If Windows shows *"Windows protected your PC"*, click **More info → Run anyway**.
3. Wait until the window says **"Setup finished"** (Python, packages, models, speech model ~460 MB, summary model ~5 GB). Don't close it. If it fails, run it again — that is safe.

## Step 4 — Start the app

1. Double-click **`run_app.bat`**.
2. The browser opens at **http://127.0.0.1:8765**. Click **Allow** for camera and microphone.
3. Keep the black window open while using it. Closing it stops the app.

Next time, only Step 4 is needed.

## Using the app

- **Patient:** answer the questions and do the four tasks.
- **Doctor:** click **"New doctor? Sign up"**, choose your own username and password, and **save the recovery code**.
- Records are saved on **your laptop** (`data\local_store.json`) unless you set up your own Supabase (see `03_database.md`). For that, copy `.env.example` to a new file named `.env` and fill in **your own** keys. Never share your keys.

## Getting updates later

- If you used **Download ZIP**: download the new zip again and replace the old folder (keep your `models` folder and `data` folder).
- If you used **git clone**: open PowerShell in the folder and type `git pull`.

## If something goes wrong

| Problem | What to do |
|---|---|
| "Models still missing" | `models.zip` is not in the project folder — repeat Step 2, then run `setup_new_pc.bat` again |
| "Python environment not found" | Run `setup_new_pc.bat` first |
| Summary is plain fixed text | The 5 GB summary model isn't installed. The app still works |
| Eye test: face not tracked | Face the camera in good light, keep still |
| Page doesn't open | Use Chrome/Edge and exactly `http://127.0.0.1:8765` |

To understand how the app works, read the other files in this `project` folder, starting with `README.md`.
