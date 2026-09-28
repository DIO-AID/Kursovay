#!/usr/bin/env bash
# Установка окружения на Linux/macOS, запуск из папки diploma_fs:  bash scripts/setup.sh
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
if [ -f requirements.lock.txt ]; then
  $PY -m pip install -r requirements.lock.txt
else
  $PY -m pip install -r requirements.txt
  $PY -m pip freeze > requirements.lock.txt
fi
$PY scripts/check_env.py
echo -e "\nГотово. Активировать окружение: source .venv/bin/activate"
