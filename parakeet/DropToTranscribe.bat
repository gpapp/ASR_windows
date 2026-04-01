@echo off
setlocal enabledelayedexpansion

:: 1. Check if venv exists
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Error: Virtual environment not found. Run setup_env.bat first.
    pause
    exit /b
)

:: 2. Check if Server is TRULY responsive (Health Check)
powershell -command "try { $response = Invoke-WebRequest -Uri http://127.0.0.1:8000/health -UseBasicParsing -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1

if %errorlevel% == 0 (
    echo [SYSTEM] Server is already running. Using existing instance.
    goto process_start
)

echo [SYSTEM] Server not responding. Starting Nemotron Server...

:: Launch server in a new minimized window
start /min "Parakeet Server" cmd /c "call "%~dp0.venv\Scripts\activate" && python "%~dp0server.py""

echo [SYSTEM] Waiting for model to load in RAM...

:wait_loop
timeout /t 3 /nobreak >nul
powershell -command "try { $response = Invoke-WebRequest -Uri http://127.0.0.1:8000/health -UseBasicParsing -TimeoutSec 1; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    goto wait_loop
)
echo [SYSTEM] Server is now ONLINE.

:process_start

:: 3. Run the transcription
if "%~1"=="" (
    echo [INFO] No files dropped.
    echo [INFO] Supported formats: .mp4, .mp3, .wav, .m4a, .flac, .mov, .mkv
    echo [INFO] Server is ready and waiting.
    pause
    exit /b
)

echo [INFO] Starting Transcription Pipeline...
"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe.py" %*

echo.
echo ------------------------------------------
echo Processing Complete.
pause