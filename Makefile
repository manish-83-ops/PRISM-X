# =============================================================================
# PRISMX Master Automation Makefile
#
# STORAGE VOLUME PATH REQUIREMENTS:
# In Docker environments, persistent volumes must use named Docker volumes
# (prismx_qdrant_storage, prismx_sqlite_data) which reside on native filesystem
# (e.g., /var/lib/docker/volumes or WSL2 ext4) OUTSIDE OneDrive/cloud sync paths.
# Local execution should place data/ outside any synchronized folders.
# =============================================================================

.PHONY: demo test bench eval-smoke snapshot-restore lint verify-numbers clean

# Run local interactive demo verification
demo:
	python scripts/predemo_check.py

# Run unit tests (excluding tests requiring 100K store)
test:
	pytest -m "not needs_100k" -v --tb=short

# Run full HTTP latency benchmark (100 BENCH queries across dense/hybrid/prismx)
bench:
	python scripts/run_latency_benchmark_gate5.py --mode all --n-queries 100

# Run evaluation smoke test across retrieval and anytime governor components
eval-smoke:
	pytest tests/test_service.py tests/test_fusion.py tests/test_reranker_cascade.py tests/test_filters.py -v

# Restore Qdrant c100k_raw snapshot from snapshots/ directory
snapshot-restore:
	python scripts/restore_snapshot.py

# Run secrets safety scanner
lint:
	python scripts/check_secrets.py

# Recompute and verify all headline report and README numbers
verify-numbers:
	python scripts/verify_report_numbers.py

# Clean cache and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
