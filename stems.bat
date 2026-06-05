@echo off
rem ---------------------------------------------------------------------------
rem  stems.bat - run the stems CLI inside the project venv without having to
rem  activate it yourself. Activates the venv, runs stems with whatever args
rem  you pass, then deactivates. The CLI's exit code is preserved.
rem
rem  Usage (from anywhere):
rem      stems.bat separate "song.webm" -p vocals-max
rem      stems.bat presets
rem ---------------------------------------------------------------------------
setlocal

set "VENV=%~dp0.venv\Scripts"

if not exist "%VENV%\activate.bat" (
    echo [stems.bat] venv not found at "%VENV%". Create it first.
    exit /b 1
)

call "%VENV%\activate.bat"
rem Call the venv's stems.exe by full path so this never re-invokes stems.bat.
"%VENV%\stems.exe" %*
set "EXITCODE=%ERRORLEVEL%"
call "%VENV%\deactivate.bat"

endlocal & exit /b %EXITCODE%
