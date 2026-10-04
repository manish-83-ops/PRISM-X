# PRISMX Experiments Archive

This directory contains standalone investigative scripts, exploratory analysis tools, and retired experimental pipelines created during intermediate design gates.

None of the scripts in this directory are in the active production serving, testing, or benchmark reproduction path. All empirical results remain permanently recorded in `results/`.

## Directory Contents

| Script / Directory | Gate / Phase | Description | Status |
| :--- | :--- | :--- | :--- |
| `debug_stress_anomaly.py` | Gate 5.2 | Investigated graph traversal anomalies during distractor stress testing on `c100k_hard`. | Retired (ADR-015 superseded by `c100k_raw`) |
| `measure_reserve.py` | Gate 4A | Diagnostic script measuring CPU thread headroom and RAM allocation under concurrent load. | Completed diagnostic |
| `exp_001_ingest/` | Gate 1 | Initial exploratory ingestion trial on 10k MS MARCO sample. | Preserved baseline |
| `INDEX.csv` | Gate 1-5 | Historical index of pre-registered experiment IDs and outcomes. | Maintained registry |
