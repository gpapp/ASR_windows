@echo off
setlocal

:: Check if venv exists
if not exist "%~dp0venv\Scripts\python.exe" (
    echo Error: Virtual environment not found. 
    echo Please run setup_env.bat first.
    pause
    exit /b
)

if "%~1"=="" (
    echo No files detected. Drag and drop video/audio files onto this icon.
    pause
    exit /b
)

echo Starting Transcription Pipeline...
:: Use the venv python directly to run the script
"%~dp0venv\Scripts\python.exe" "%~dp0transcribe.py" %*

echo.
echo ------------------------------------------
echo Processing Complete.
pause