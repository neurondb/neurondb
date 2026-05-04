# Performance baselines

Goal: catch large regressions in vector distance, index build, and small ML training paths without flaking CI.

## Approach

1. Choose **5–10** stable scripts from `src/tests/sql` or short custom `psql` timings.
2. Run them on a **dedicated** runner (same CPU class, cold vs warm cache documented).
3. Store **JSON** summaries next to the nightly workflow artifact (median wall time per step).
4. Alert when a metric exceeds **X%** above a rolling median (configure X per team).

## Placeholder

Automation can extend [scripts/perf_smoke.sh](../../scripts/perf_smoke.sh) to emit machine-readable timings for ingestion by CI.
