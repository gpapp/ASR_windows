@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

:: Default options - use voiceprints from parent folder if exists
set DEFAULT_ARGS=--diarization-threshold 0.7
set VOICEPRINTS_ARG=

if exist "voiceprints.json" (
    set VOICEPRINTS_ARG=--voiceprints "%~dp0voiceprints.json"
) else if exist "..\voiceprints.json" (
    set VOICEPRINTS_ARG=--voiceprints "..\voiceprints.json"
)

if "%~1"=="" (
    echo Drop audio or video files onto this batch file to transcribe them.
    echo Supported formats: .mp3 .mp4 .wav .m4a .flac .mov .mkv .avi .webm .ogg
    echo.
    echo Using voiceprints for speaker recognition if available.
    echo Default: 4 speakers, strict clustering threshold
    pause
    exit /b 0
)

:: Check if server is already running and ready
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2; $j = $r.Content | ConvertFrom-Json; if ($j.model_status -eq 'ready') { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>&1
set SERVER_STATUS=%errorlevel%
if %SERVER_STATUS% equ 0 goto run

:: Server not running or not yet ready — start it only if not running at all
if %SERVER_STATUS% equ 1 (
    echo [INFO] Starting transcription server...
    set LAUNCH_DIR=%~dp0
    start "Cohere Transcribe Server" /min cmd /c "cd /d ""%~dp0"" && call .venv\Scripts\activate && python server.py"
)

:: Poll until model is ready (first run downloads ~2.9 GB, so be patient)
echo [INFO] Waiting for server to be ready (first run downloads ~2.9 GB)...
:wait_loop
timeout /t 3 /nobreak >nul
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2; $j = $r.Content | ConvertFrom-Json; if ($j.model_status -eq 'ready') { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 goto wait_loop

:run
echo [INFO] Server ready. Transcribing with voiceprints (4 speakers, threshold 0.2)...
.venv\Scripts\python.exe transcribe.py %DEFAULT_ARGS% %VOICEPRINTS_ARG% %*

echo.
echo [INFO] Done.
pause
