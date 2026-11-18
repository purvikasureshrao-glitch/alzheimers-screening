# Friend guide: run the app from the ZIP (no Git, no accounts)

You need: a Windows 10/11 laptop, internet for the first setup (about 6 GB, 30 minutes), a webcam and a microphone.

1. **Download** `Alzheimers_for_team.zip` (sent on WhatsApp as a document).
2. **Extract it:** right-click the zip, choose **Extract All**, and pick `C:\` so you get `C:\Alzheimers`. Avoid the Downloads folder and long paths.
3. **One-time setup:** open the folder and double-click **`setup_new_pc.bat`**.
   - If Windows says "protected your PC", click **More info**, then **Run anyway**.
   - Say **Yes** to any install prompt.
   - Wait for the line **Setup finished**. Do not close the window before that.
4. **Start the app:** double-click **`run_app.bat`**. The browser opens at http://127.0.0.1:8765. Keep the black window open.
5. When the browser asks, click **Allow** for the **camera** and **microphone**.
6. To stop the app, close the black window.

From the next time, only step 4 is needed.

**You do NOT need** any datasets, Git, or GitHub.

| Problem | Fix |
|---|---|
| "Python environment not found" | Run `setup_new_pc.bat` again |
| Setup fails at a download | Check the internet and run `setup_new_pc.bat` again. It resumes |
| Summary is plain template text | The Ollama download is still running or failed. The app still works |
