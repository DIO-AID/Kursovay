# Установка окружения на Windows (PowerShell), запуск из папки diploma_fs:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # только ядро
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Extra   # + catboost/shap/optuna/streamlit
param([switch]$Extra)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
if (Test-Path "requirements.lock.txt") {
    & $py -m pip install -r requirements.lock.txt
} else {
    & $py -m pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw "Установка ядра не удалась - пришлите последние 30 строк вывода" }
if (-not (Test-Path "requirements.lock.txt")) {
    # ASCII: pip читает lock-файл без проблем с BOM/UTF-16
    & $py -m pip freeze | Out-File -Encoding ascii requirements.lock.txt
}
if ($Extra) {
    & $py -m pip install -r requirements-extra.txt
    if ($LASTEXITCODE -ne 0) { Write-Warning "Дополнительные пакеты не встали - ядро работает, пришлите лог" }
}
& $py scripts\check_env.py
Write-Host "`nГотово. Активировать окружение: .\.venv\Scripts\Activate.ps1"
