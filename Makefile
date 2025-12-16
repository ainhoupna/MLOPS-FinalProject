.PHONY: install lint format test run clean

install:
	uv sync

lint:
	uv run pylint src app tests

format:
	uv run black .

test:
	uv run pytest tests/ -v

run:
	uv run uvicorn src.api.main:app --reload

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf mlruns
	rm -rf models
	rm -rf data/processed
