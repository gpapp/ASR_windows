@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Drop audio or video files onto this batch file to transcribe them.
    echo Supported formats: .mp3 .mp4 .wav .m4a .flac .mov .mkv .avi .webm .ogg
    pause
    exit /b 0
)

set "BASE_MODEL=gemma4:e2b"
set "CTX_SIZE=32768"
set "CUSTOM_MODEL_NAME=%BASE_MODEL%-summarizer"

:: --- STEP 1: CREATE CUSTOM MODEL ---
ollama list | findstr /C:"%CUSTOM_MODEL_NAME%" >nul
if %errorlevel% neq 0 (
    echo [SETUP] Creating custom model...
    (
        echo FROM %BASE_MODEL%
        echo PARAMETER num_ctx %CTX_SIZE%
        echo PARAMETER temperature 1.0
    ) > Modelfile
    ollama create %CUSTOM_MODEL_NAME% -f Modelfile >nul 2>&1
    if exist Modelfile del Modelfile
    echo [SETUP] Model created.
)

:: Check if server is already running and ready
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/' -UseBasicParsing -TimeoutSec 2; if ($r.Content -eq 'Ollama is running') { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% equ 0 goto run

:: Server not running — start it
if %errorlevel% equ 1 (
    echo [INFO] Starting Ollama server...
    start "Cohere Transcribe Server" /min cmd /c "ollama serve"
)

:: Poll until model is ready (first run downloads ~2.9 GB, so be patient)
echo [INFO] Waiting for server to be ready (first run downloads ~2.9 GB)...
:wait_loop
timeout /t 3 /nobreak >nul
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:11434/' -UseBasicParsing -TimeoutSec 2; if ($r.Content -eq 'Ollama is running') { exit 0 } else { exit 2 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 goto wait_loop

:run
echo [INFO] Server ready. Transcribing files...
.venv\Scripts\python.exe transcribe.py --model "%CUSTOM_MODEL_NAME%" %*

echo.
echo [INFO] Done.
pause
