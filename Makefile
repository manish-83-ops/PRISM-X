# PRISMX Automation Makefile (Convenience only)

.PHONY: test lint clean run-server run-qdrant check-secrets env-report

test:
	python -m pytest tests/

lint:
	python scripts/check_secrets.py

run-qdrant:
	./bin/qdrant.exe --config-path config/qdrant.yaml

run-server:
	python -m prismx server

check-secrets:
	python scripts/check_secrets.py

env-report:
	python scripts/env_report.py
