# How to run the Alzheimer's screening app on a new Windows PC

Two parts: **A) how Git works** (short, plain words) and **B) steps for your team**.

---

## A. How Git / GitHub works (in 2 minutes)

- **Git** keeps every version of a project's files. **GitHub** is a website that stores that project online.
- The project lives in a **repository** ("repo") on GitHub. Anyone with the link can get a copy.
- You do **not** need to learn Git commands. Your team has two easy ways to get the files:
  1. **Download ZIP** (no account, no Git): open the repo page, click the green **Code** button, then **Download ZIP**, then right-click the ZIP and choose **Extract All**.
  2. **Git clone** (optional): install Git for Windows, then run `git clone <repo-link>`. The advantage is that later updates arrive with `git pull`.
- **Limit:** GitHub refuses files over 100 MB, and the trained models are 430 MB. So the models are **not** in the repo. They are shared as one extra file called `models.zip`, attached to the repo under **Releases** (step 3 below).

**Git or ZIP?** Git/GitHub carries the code, and the one big `models.zip` is downloaded once. A single ZIP of everything (about 7 GB with the speech and summary models) is a poor fit because it is too big for e-mail and most free hosts. The setup file downloads those large pieces for you.

---

## B. Steps for a teammate (needs internet, about 30 minutes, once)

**1. Get the project folder.** Use Download ZIP, then Extract All. Put the folder somewhere simple, such as `C:\Alzheimers`.

**2. Get `models.zip`.** Download it from the repo's **Releases** page (right-hand side of the repo page).
Save it **inside the project folder**, next to `run_app.bat`. Do not unzip it. Setup does that.
(If the owner pasted a direct link into `MODELS_URL.txt`, setup downloads it automatically and you can skip this step.)

**3. Double-click `setup_new_pc.bat`.** It installs Python 3.13, the packages, the speech model, Ollama and the summary model. Approve the Windows prompts. Wait until it says **Setup finished**.

**4. Double-click `run_app.bat`.** The browser opens at http://127.0.0.1:8765. Keep the black window open while using the app. Closing it stops the app.

**Allow the camera and microphone** when the browser asks. The eye test needs the webcam and the speech test needs the mic.

### If something goes wrong
| Problem | Fix |
|---|---|
| "Python environment not found" | Run `setup_new_pc.bat` first |
| "models folder is missing files" | Do step 2, then run setup again |
| Setup stops on `winget` | Install Python 3.13 from python.org (tick "Add to PATH") and Ollama from ollama.com, then run setup again |
| Summary looks like fixed template text | Ollama isn't running or the model hasn't finished downloading. The app still works |
| Windows SmartScreen warning on a `.bat` | Click **More info**, then **Run anyway** |

Records are saved in `data/local_store.json` on that PC. Cloud (Supabase) storage is off unless a `.env` file is added, which is not needed for the demo.

---

## C. For the owner: publishing steps
1. Create a free GitHub account, click **New repository**, name it, and click Create.
2. In this folder, run: `git remote add origin <repo-link>` then `git push -u origin master`.
3. On the repo page, open **Releases**, then **Create a new release**, and attach `release\models.zip`. Publish it.
4. Send your team the repo link. Test the whole flow on a second PC **before Saturday**.
