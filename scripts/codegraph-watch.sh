#!/bin/bash
# Script для watch-режима codegraph с использованием inotifywait
# Запустить: ./scripts/codegraph-watch.sh

WATCH_DIR="${1:-.}"

echo "👀 Запуск codegraph watch для: $WATCH_DIR"
echo "Нажмите Ctrl+C для выхода"

# Проверяем наличие inotifywait
if ! command -v inotifywait &> /dev/null; then
    echo "❌ inotifywait не найден. Установите: sudo pacman -S inotify-tools"
    exit 1
fi

# Основной цикл с inotifywait
inotifywait -m -r \
    --exclude '\.venv|\.git|\.cache|__pycache__|\.pyc' \
    -e modify,create,delete,move \
    --format '%w%f %e' \
    "$WATCH_DIR" | \
while read FILE EVENT; do
    if [[ "$FILE" == *.py ]]; then
        echo "📝 Изменение: $FILE ($EVENT)"
        codegraph sync 2>/dev/null || true
    fi
done
