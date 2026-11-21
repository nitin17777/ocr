# Single shortcut runner for OCR LoRA Project
param(
    [Parameter(Position=0)]
    [string]$Command = "web",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PSScriptRoot

# Activate virtual environment
$VenvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}
$env:PYTHONPATH = $PSScriptRoot

switch ($Command.ToLower()) {
    "web" {
        Write-Host "Starting OCR LoRA Web Server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
        & $VenvPython -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000 @RemainingArgs
    }
    "infer" {
        & $VenvPython -m ocr_lora.infer @RemainingArgs
    }
    "eval" {
        & $VenvPython -m ocr_lora.evaluate @RemainingArgs
    }
    "train" {
        & $VenvPython train.py @RemainingArgs
    }
    "test" {
        & $VenvPython -m pytest @RemainingArgs
    }
    default {
        # Fallback to python with arguments
        & $VenvPython $Command @RemainingArgs
    }
}
