.PHONY: test integration check build

test:
	uv run pytest -q --tb=short

integration:
	uv run pytest tests/test_database_integration.py -q --tb=short

check:
	uv run python -m compileall -q corpus_api smart_agents learning_classifier learning_corpus_pipeline
	uv run python scripts/check_docs.py

build:
	uv build
