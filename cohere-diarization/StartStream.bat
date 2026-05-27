@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

:: Parse output file argument
set "OUTPUT_FILE=%~1"
if "!OUTPUT_FILE!"=="" (
    for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"') do set "DT=%%i"
    set "OUTPUT_FILE=stream_!DT!.txt"
)

:: Check if server is already running and ready
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2; $j = $r.Content | ConvertFrom-Json; if ($j.model_status -eq 'ready') { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>&1
set SERVER_STATUS=%errorlevel%
if %SERVER_STATUS% equ 0 goto run

:: Server not running or not yet ready — start it
if %SERVER_STATUS% equ 1 (
    echo [INFO] Starting transcription server...
    set "LAUNCH_DIR=%~dp0"
    set "LAUNCH_DIR=!LAUNCH_DIR:~0,-1!"
    start "Cohere Transcribe Server" /min cmd /c "cd /d "!LAUNCH_DIR!" && call .venv\Scripts\activate && python server.py"
)

:: Poll until model is ready (first run downloads ~2.9 GB)
echo [INFO] Waiting for server to be ready (first run downloads ~2.9 GB)...
:wait_loop
timeout /t 3 /nobreak >nul
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2; $j = $r.Content | ConvertFrom-Json; if ($j.model_status -eq 'ready') { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 goto wait_loop

:run
echo [INFO] Server ready. Starting stream capture...
echo [INFO] Output: !OUTPUT_FILE!
echo [INFO] Press Ctrl+C to stop recording.
echo.
.venv\Scripts\python.exe stream_client.py "!OUTPUT_FILE!"

echo.
echo [INFO] Done.
pause
