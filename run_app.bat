@echo off
rem Starts everything: Ollama (for the written summary), the screening app, and the browser. Close this window to stop the app.
cd /d "%~dp0"
set PYTHONPATH=.
set PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
if not exist "%PY%" ( echo Python 3.13 not found at %PY% & pause & exit /b 1 )
tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul || start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
rem open the browser a few seconds later, once the server is listening
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8765"
echo Alzheimer's screening app: http://127.0.0.1:8765   (Ctrl+C or close this window to stop)
"%PY%" -m app.server
pause
