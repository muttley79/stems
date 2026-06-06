@echo off
rem ---------------------------------------------------------------------------
rem  stems-gui.bat - launch the stems desktop GUI from the project venv.
rem
rem  stems-gui.exe is a gui-script (pythonw) whose launcher already embeds the
rem  venv interpreter, so there is no need to activate the venv. We just `start`
rem  it detached and exit immediately, so this cmd window closes at once instead
rem  of lingering. For a truly flash-free launch, make a shortcut straight to
rem  .venv\Scripts\stems-gui.exe (no console host at all).
rem
rem  Usage (from anywhere):
rem      stems-gui.bat
rem ---------------------------------------------------------------------------
setlocal

set "GUI=%~dp0.venv\Scripts\stems-gui.exe"

if not exist "%GUI%" (
    echo [stems-gui.bat] stems-gui not installed. Run: pip install -e .[gui]
    exit /b 1
)

start "" "%GUI%" %*

endlocal & exit /b 0
