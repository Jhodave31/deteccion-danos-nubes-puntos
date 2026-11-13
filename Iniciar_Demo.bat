@echo off
chcp 65001 >nul
title Demo - Deteccion de danos estructurales
cd /d "%~dp0demo"
echo Iniciando la demo... se abrira el navegador en http://localhost:8000
echo (Para cerrar la demo, cierre esta ventana.)
echo.
set "PY=python"
if exist "%USERPROFILE%\.venvs\electiva\Scripts\python.exe" set "PY=%USERPROFILE%\.venvs\electiva\Scripts\python.exe"
"%PY%" app.py
pause
