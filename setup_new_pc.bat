@echo off
rem ONE-TIME setup for a new Windows PC. Needs internet. Takes 15-40 minutes (mostly downloads). Safe to run again.
cd /d "%~dp0"
echo ============================================================
echo  Alzheimer's screening app - one-time setup
echo ============================================================

rem ---- 1. Python 3.13 -------------------------------------------------------
set BASEPY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
if not exist "%BASEPY%" (
  echo [1/5] Installing Python 3.13 ...
  winget install -e --id Python.Python.3.13 --scope user --accept-package-agreements --accept-source-agreements
)
if not exist "%BASEPY%" (
  echo Could not install Python automatically. Install Python 3.13 from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this file again.
  pause
  exit /b 1
)
echo [1/5] Python OK

rem ---- 2. Packages ----------------------------------------------------------
echo [2/5] Installing packages ...
if not exist ".venv\Scripts\python.exe" "%BASEPY%" -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo Package install failed - check the internet connection and run again. & pause & exit /b 1 )

rem ---- 3. Trained models ----------------------------------------------------
if exist "models\clock_vit.onnx" (
  echo [3/5] Models already present
) else (
  echo [3/5] Getting the trained models ...
  if exist "models.zip" (
    powershell -NoProfile -Command "Expand-Archive -Force 'models.zip' '.'"
  ) else (
    if not exist "MODELS_URL.txt" ( echo MODELS_URL.txt is missing. See project\10_friend_zip_guide.md. & pause & exit /b 1 )
    set /p MODELS_URL=<MODELS_URL.txt
    call powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri (Get-Content MODELS_URL.txt -TotalCount 1) -OutFile models.zip; Expand-Archive -Force models.zip ."
  )
)
if not exist "models\clock_vit.onnx" ( echo Models still missing - See project\10_friend_zip_guide.md. & pause & exit /b 1 )

rem ---- 4. Speech recognition model (downloads once, ~460 MB) -----------------
echo [4/5] Downloading the speech model ...
".venv\Scripts\python.exe" -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('speech model ready')"

rem ---- 5. Ollama + language model for the written summary (~5 GB) -------------
set OLLAMA=
where ollama >nul 2>nul && set OLLAMA=ollama
if "%OLLAMA%"=="" if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
if "%OLLAMA%"=="" (
  echo [5/5] Installing Ollama ...
  winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
  if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
)
if "%OLLAMA%"=="" (
  echo Ollama could not be installed. The app still works; the summary will use the built-in template text.
) else (
  tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul || ( start "" /min "%OLLAMA%" serve & timeout /t 6 /nobreak >nul )
  echo [5/5] Downloading the summary model ^(about 5 GB, be patient^) ...
  "%OLLAMA%" pull llama3.1:8b
)

echo.
echo ============================================================
echo  Setup finished. From now on just double-click run_app.bat
echo ============================================================
pause
