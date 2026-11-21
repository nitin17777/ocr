@echo off
setlocal
cd /d "%~dp0"
call ".\venv\Scripts\activate.bat"
set PYTHONPATH=.

:: Default action (no arguments): launch web app
if "%~1"=="" (
    echo Starting OCR LoRA Web Server on http://127.0.0.1:8000 ...
    python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
    goto :eof
)

:: Subcommands
if /i "%~1"=="web" (
    python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
    goto :eof
)

if /i "%~1"=="infer" (
    shift
    python -m ocr_lora.infer %1 %2 %3 %4 %5 %6 %7 %8 %9
    goto :eof
)

if /i "%~1"=="eval" (
    shift
    python -m ocr_lora.evaluate %1 %2 %3 %4 %5 %6 %7 %8 %9
    goto :eof
)

if /i "%~1"=="train" (
    shift
    python train.py %1 %2 %3 %4 %5 %6 %7 %8 %9
    goto :eof
)

if /i "%~1"=="test" (
    shift
    pytest %1 %2 %3 %4 %5 %6 %7 %8 %9
    goto :eof
)

:: Fallback: pass-through to python
python %*
endlocal
