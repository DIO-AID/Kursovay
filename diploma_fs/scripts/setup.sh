#!/usr/bin/env bash
# Установка на Linux/macOS, из папки diploma_fs:  bash scripts/setup.sh [--extra]
set -e
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip setuptools wheel
if [ -f requirements.lock.txt ]; then
  $PY -m pip install -r requirements.lock.txt
else
  $PY -m pip install -r requirements.txt
  $PY -m pip freeze > requirements.lock.txt
fi
if [ "$1" = "--extra" ]; then
  $PY -m pip install -r requirements-extra.txt || echo "!! дополнительные пакеты не встали, ядро работает"
fi
$PY scripts/check_env.py
echo -e "\nГотово. Активировать окружение: source .venv/bin/activate"
