# Установка окружения на Windows (PowerShell), запуск из папки diploma_fs:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if (Test-Path "requirements.lock.txt") {
    & $py -m pip install -r requirements.lock.txt
} else {
    & $py -m pip install -r requirements.txt
    & $py -m pip freeze | Out-File -Encoding utf8 requirements.lock.txt
}
& $py scripts\check_env.py
Write-Host "`nГотово. Активировать окружение: .\.venv\Scripts\Activate.ps1"
