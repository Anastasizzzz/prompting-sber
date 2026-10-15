#!/usr/bin/env bash
# Единая точка входа проекта.
#   ./run.sh setup                 создать .venv и поставить зависимости
#   ./run.sh test                  офлайн-тесты (без вызовов API)
#   ./run.sh check                 ключ, доступ к моделям, счётчик токенов
#   ./run.sh data                  обзор данных и золотого набора (без API)
#   ./run.sh index                 индекс по raw_data/SPIEF_txt (один раз, ≈50 ₽)
#   ./run.sh ask "вопрос" [--config A|FINAL|...] [--cache]
#   ./run.sh eval --config FINAL --cases core [--repeats 2]
#   ./run.sh experiments [имена...] [--budget 3000]
#   ./run.sh report                results/REPORT.md
#   ./run.sh demo [--no-pause] [--steps 2,5]
#   ./run.sh all                   index + experiments + report
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV=".venv"

setup() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN=python3
  fi
  "$PYTHON_BIN" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements.txt
  echo "Окружение готово: $VENV"
}

run_py() {
  if [ ! -x "$VENV/bin/python" ]; then
    setup
  fi
  PYTHONUNBUFFERED=1 PYTHONPATH=src "$VENV/bin/python" -m agent "$@"
}

command="${1:-help}"
shift || true

case "$command" in
  setup) setup ;;
  test)
    [ -x "$VENV/bin/python" ] || setup
    PYTHONPATH=src "$VENV/bin/python" -m pytest -q tests "$@"
    ;;
  check|data|index|ask|eval|experiments|report|demo) run_py "$command" "$@" ;;
  all)
    run_py index
    run_py experiments "$@"
    run_py report
    ;;
  help|-h|--help) sed -n '2,15p' "$0" ;;
  *) echo "Неизвестная команда: $command"; sed -n '2,15p' "$0"; exit 1 ;;
esac
