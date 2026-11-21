@echo off
setlocal
cd /d "%~dp0"
echo Building Tailwind CSS for OCR LoRA Webapp...

set CLI_PATH="..\..\tailwindcss.exe"

if exist %CLI_PATH% (
    %CLI_PATH% -c tailwind.config.js -i input.css -o css/tailwind.css --minify
) else (
    echo Local tailwindcss.exe not found at root, attempting global tailwindcss...
    tailwindcss -c tailwind.config.js -i input.css -o css/tailwind.css --minify
)

if %ERRORLEVEL% equ 0 (
    echo [SUCCESS] Tailwind CSS compiled to webapp\static\css\tailwind.css
) else (
    echo [ERROR] Failed to compile Tailwind CSS.
)

endlocal
