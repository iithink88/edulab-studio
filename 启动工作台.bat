@echo off
rem edulab Studio launcher (ASCII only; chcp 65001 for UTF-8 console output)
setlocal
cd /d %~dp0
set PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe
if not exist "%PY%" set PY=python
echo ============================================
echo   edulab one-click courseware studio
echo   Starting local server... browser opens automatically
echo ============================================
"%PY%" edulab_studio.py
if errorlevel 1 pause
