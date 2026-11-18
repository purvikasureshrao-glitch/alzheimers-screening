# Friend guide: get the app from GitHub

Replace `YOUR-USERNAME/YOUR-REPO` with the real repo (the owner will tell you).

## Easiest: Download ZIP (no Git, no account)
1. Open `https://github.com/YOUR-USERNAME/YOUR-REPO`.
2. Click the green **Code** button, then **Download ZIP**.
3. Right-click the ZIP, choose **Extract All**, and extract to `C:\Alzheimers`.
4. On the repo page, click **Releases** (right side), open the latest release, and download **`models.zip`**.
5. Put `models.zip` **inside** `C:\Alzheimers`, next to `run_app.bat`. Do NOT unzip it.
6. Double-click **`setup_new_pc.bat`** (once, about 30 minutes, needs internet). Wait for "Setup finished".
7. Double-click **`run_app.bat`**. The browser opens the app. Allow camera and microphone.

## Alternative: with Git (gets later updates)
Open PowerShell and run:
```
winget install Git.Git
```
Close and reopen PowerShell, then:
```
cd C:\
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git Alzheimers
cd Alzheimers
```
Then do steps 4 to 7 above (download `models.zip` into the folder, run setup, run the app).
To get the owner's later changes: `cd C:\Alzheimers` then `git pull`.

## How Git works in one minute
- A **repository** is the project folder stored online on GitHub.
- **clone** copies it to your PC. **pull** brings new changes. **push** sends your changes (the owner does this).
- Big trained models can't go in Git, so they are in **Releases** as `models.zip`.
