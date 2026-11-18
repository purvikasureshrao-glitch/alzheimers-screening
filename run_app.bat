@echo off
rem Starts everything: Ollama (for the written summary), the screening app, and the browser. Close this window to stop the app.
rem First time on a new PC? Run setup_new_pc.bat once (see HOW_TO_RUN.md).
cd /d "%~dp0"
set PYTHONPATH=.
set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
if not exist "%PY%" (
  echo Python environment not found. Please double-click setup_new_pc.bat first ^(one time only^).
  pause
  exit /b 1
)
if not exist "models\clock_vit.onnx" (
  echo The models folder is missing files. Run setup_new_pc.bat, or see HOW_TO_RUN.md step 3.
  pause
  exit /b 1
)
set OLLAMA=
where ollama >nul 2>nul && set OLLAMA=ollama
if "%OLLAMA%"=="" if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
if "%OLLAMA%"=="" (
  echo Note: Ollama is not installed, so the written summary will use the built-in template text.
) else (
  tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul || start "" /min "%OLLAMA%" serve
)
rem open the browser a few seconds later, once the server is listening
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8765"
echo Alzheimer's screening app: http://127.0.0.1:8765   (Ctrl+C or close this window to stop)
"%PY%" -m app.server
pause
