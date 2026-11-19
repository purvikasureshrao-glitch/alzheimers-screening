# How to use the ZIP file (for the person who received it)

You were sent a file called **Alzheimers_1.0.zip** on WhatsApp or Telegram. This guide gets the app running on your laptop.
No programming knowledge is needed.

> This is a research / student screening aid. It does **not** diagnose anything.

## What you need

- A **Windows 10 or 11** laptop (Mac and Linux are not supported)
- About **10 GB** of free disk space
- **Internet** for the first setup only (15–40 minutes, mostly downloads)
- A **webcam** and a **microphone**
- **Chrome** or **Edge**

## Step 1 — Save and unzip

1. Save the zip from WhatsApp/Telegram to your computer (for example into **Downloads**).
2. Right-click the zip → **Extract All…** → choose a simple place such as `C:\Alzheimers_1.0` → **Extract**.
   Do not double-click files while they are still inside the zip.
3. Open the extracted folder. You should see `app`, `models`, `project`, `run_app.bat`, `setup_new_pc.bat` and more.

## Step 2 — One-time setup

1. Double-click **`setup_new_pc.bat`**.
2. If Windows shows a blue box saying *"Windows protected your PC"*, click **More info → Run anyway**. (It is a plain text file; you can open it in Notepad to read it.)
3. A black window opens and works through 5 steps: Python, packages, models, speech model (~460 MB), and the summary language model (~5 GB).
4. **Wait** until you see **"Setup finished"**. This takes a while. Do not close the window.
5. If it stops with an error, check your internet and double-click `setup_new_pc.bat` again — it is safe to repeat.

You only do Step 2 once.

## Step 3 — Start the app

1. Double-click **`run_app.bat`**.
2. Your browser opens at **http://127.0.0.1:8765**. (If not, open Chrome and type that address.)
3. Keep the black window open while you use the app. **Closing it stops the app.**
4. When the browser asks for **camera** and **microphone**, click **Allow**.

Next time, you only need Step 3.

## Using the app

- **"I'm a patient":** fill in the questions, then do the four tasks (clock, finger tapping, eye movement, speech). About 10 minutes.
- **"I'm a doctor":** click **"New doctor? Sign up"**, choose your own username and password, and **write down the recovery code** it shows — it is the only way to reset a forgotten password. Then you can search a patient and see their results.
- Nobody has to give you a password or key. You make your own account.

## Where are the records saved?

On **your own laptop**, in the file `data\local_store.json` inside the folder. Nothing is sent to the person who gave you the zip.

*Optional:* to keep records online, create your own free Supabase project (see `03_database.md`) and put your own keys in a file named `.env` (copy `.env.example`). **Never** ask for someone else's keys and never share yours.

## If something goes wrong

| Problem | What to do |
|---|---|
| "Python environment not found" | Run `setup_new_pc.bat` first |
| A banner says model files are missing | The unzip was incomplete — extract the zip again |
| Summary looks like plain, fixed text | The 5 GB summary model isn't installed. The app still works |
| First clock or speech test is slow | Normal — models load the first time |
| Eye test says your face was not tracked | Face the camera, good light, keep still |
| Page doesn't open | Use Chrome or Edge, address exactly `http://127.0.0.1:8765` |

To understand how the app works, read the other files in this `project` folder, starting with `README.md`.
