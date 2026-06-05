@echo off
rem ---------------------------------------------------------------------------
rem  stems-gui.bat - launch the stems desktop GUI inside the project venv
rem  without having to activate it yourself. Mirrors stems.bat. Because the GUI
rem  is a gui-script (pythonw), no console window is attached.
rem
rem  Usage (from anywhere):
rem      stems-gui.bat
rem ---------------------------------------------------------------------------
setlocal

set "VENV=%~dp0.venv\Scripts"

if not exist "%VENV%\activate.bat" (
    echo [stems-gui.bat] venv not found at "%VENV%". Create it first.
    exit /b 1
)

if not exist "%VENV%\stems-gui.exe" (
    echo [stems-gui.bat] stems-gui not installed. Run: pip install -e .[gui]
    exit /b 1
)

call "%VENV%\activate.bat"
rem Launch the GUI by full path so this never re-invokes stems-gui.bat.
start "" "%VENV%\stems-gui.exe" %*
call "%VENV%\deactivate.bat"

endlocal & exit /b 0
