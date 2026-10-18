@echo off
rem Starts the screening app with Python 3.13 and opens it in the browser. Ollama (optional) must be running for the written summary.
cd /d "%~dp0"
set PYTHONPATH=.
start "" http://127.0.0.1:8765
"%LOCALAPPDATA%\Programs\Python\Python313\python.exe" -m app.server
