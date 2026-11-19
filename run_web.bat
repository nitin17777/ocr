@echo off
setlocal
cd /d "%~dp0"
call ".\venv\Scripts\activate.bat"
set PYTHONPATH=.
echo Starting OCR LoRA Correction Web Server on http://127.0.0.1:8000 ...
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
endlocal
