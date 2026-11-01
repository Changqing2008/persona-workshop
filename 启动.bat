@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Personal Workshop

rem Exit codes from run.py:
rem   0 = ran and stopped normally
rem   2 = config problem
rem   3 = another workshop is already running (browser opened it)
rem   4 = console could not start (port taken)
rem  NOTE: keep comments ASCII-only. Chinese text in rem lines breaks cmd
rem  parsing after chcp 65001.

where python >nul 2>nul
if errorlevel 1 goto NO_PY

python run.py %*
set RC=%ERRORLEVEL%
if "%RC%"=="0" exit /b 0
if "%RC%"=="3" goto ALREADY
if "%RC%"=="4" goto NO_PORT
goto FAILED

:ALREADY
echo   An instance is already running. The browser should have opened it.
echo   This window can be closed.
rem Absolute path: a Git/MSYS "timeout" on PATH would break this line.
%SystemRoot%\System32\timeout.exe /t 8 /nobreak >nul 2>nul
exit /b 0

:NO_PORT
echo.
echo   The console could not start: the port is taken by another program.
echo   Try:  python run.py --port 8900
echo.
pause
exit /b 1

:NO_PY
echo.
echo   [X] Python not found in PATH.
echo       Install Python 3.11+ from https://www.python.org/downloads/
echo       During setup, check: Add Python to PATH
echo.
pause
exit /b 1

:FAILED
echo.
echo   Startup failed. Read the message above.
echo.
pause
exit /b 1
