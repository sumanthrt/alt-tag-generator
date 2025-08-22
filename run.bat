@echo off
setlocal
chcp 65001 >nul
title Alt Tag Generator

REM Move to the folder this .bat lives in
pushd "%~dp0"

echo ===========================
echo Running Alt Tag Generator
echo ===========================

REM Ensure venv exists
if not exist "venv\Scripts\activate" (
  echo [!] Virtual environment not found.
  echo     Please run setup.bat first.
  popd
  pause
  exit /b 1
)

REM Activate venv
call "venv\Scripts\activate"

REM Run with unbuffered output for clean tqdm rendering
python -u alt_tag_generator.py
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% NEQ 0 (
  echo [!] Script exited with code %EXITCODE%.
) else (
  echo [OK] Finished successfully.
)

popd
pause
endlocal
