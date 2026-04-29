.PHONY: test test-fast lint format typecheck clean

test:
	pytest tests/

test-fast:
	pytest tests/ -m "not slow" -x

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/

typecheck:
	mypy --strict src/transfermtl

lock-check:
	python -m transfermtl.utils.config --check-lock

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
