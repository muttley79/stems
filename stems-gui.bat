@echo off
rem ---------------------------------------------------------------------------
rem  stems-gui.bat - launch the stems desktop GUI from the project venv.
rem
rem  Self-bootstrapping: on the first run (or after a new GUI dependency is
rem  added) it installs the GUI extras automatically, so you never have to run
rem  `pip install` by hand. After that it just launches and exits.
rem
rem  The window is started with pythonw (no console) and detached, so no black
rem  console window is left behind and ffmpeg/backend children stay hidden too.
rem ---------------------------------------------------------------------------
setlocal

set "ROOT=%~dp0"
set "VPY=%ROOT%.venv\Scripts\python.exe"
set "VPYW=%ROOT%.venv\Scripts\pythonw.exe"

if not exist "%VPY%" (
    echo [stems-gui] venv not found at "%ROOT%.venv".
    echo            Create it first ^(see README: py -3.10 -m venv .venv^).
    pause
    exit /b 1
)

rem Are the GUI deps + editable package present? Install once if not.
"%VPY%" -c "import customtkinter, tkinterdnd2, stems.gui" 1>nul 2>nul
if errorlevel 1 (
    echo [stems-gui] Installing GUI dependencies ^(first run^)...
    pushd "%ROOT%"
    "%VPY%" -m pip install -e .[gui]
    popd
    if errorlevel 1 (
        echo [stems-gui] Dependency install failed. See the output above.
        pause
        exit /b 1
    )
)

rem Launch windowless and detached; this cmd window closes immediately.
start "" "%VPYW%" -m stems.gui %*

endlocal & exit /b 0
