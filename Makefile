.PHONY: codegraph-sync codegraph-index codegraph-status codegraph-init help

# По умолчанию показываем справку
help:
	@echo "Доступные команды:"
	@echo "  make codegraph-sync   - Инкрементальная синхронизация codegraph индекса"
	@echo "  make codegraph-index  - Полная переиндексация"
	@echo "  make codegraph-status - Проверка состояния индекса"
	@echo "  make codegraph-init   - Инициализация codegraph"

codegraph-sync:
	codegraph sync

codegraph-index:
	codegraph index

codegraph-status:
	codegraph status

codegraph-init:
	codegraph init
