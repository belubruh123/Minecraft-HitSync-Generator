@echo off
rem Hit-Sync for Windows: double-click. The first run sets everything up.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up Hit-Sync for the first time. This takes a few minutes...
    where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
)
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python 3.10 or newer is needed. Get it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during the install. Then run this again.
    pause
    exit /b 1
)
fc /b requirements.txt .venv\requirements.installed >nul 2>nul
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing the requirements failed - see the messages above.
        pause
        exit /b 1
    )
    copy /y requirements.txt .venv\requirements.installed >nul
)
start "" ".venv\Scripts\pythonw.exe" -m hitsync %*
