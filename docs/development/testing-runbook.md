# NeuronDB testing runbook

Quick reference for contributors: which command runs which tests, and how CI maps to the repo.

## Extension (C/SQL)

- **PgXS smoke:** `make installcheck` — runs `REGRESS` in `Makefile.core` (e.g. `sql/000_smoke.sql`).
- **TAP:** `make installcheck-tap` — runs `src/t/*.t` via `prove`.
- **SQL corpus:** `src/tests/run_test.py` — all or selected scripts under `src/tests/sql/`.

Requirements: PostgreSQL dev packages (`pg_config`), extension built and installed (`make install`), database reachable (often local `trust` auth for `run_test.py`).

## Python SDK (`neurondbpy`)

Path: `src/sdks/python/`.

```bash
cd src/sdks/python
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -v tests/
```

CI: `.github/workflows/build-and-test.yml` uses the same layout (`working-directory: src/sdks/python`).

## Optional: heavy ML / ONNX SQL tests

Some `src/tests/sql` scripts assume large models or long runtimes. For CI or laptops, you can introduce an explicit gate (for example **`NDB_RUN_ML_HEAVY=1`**) in `run_test.py` and document which filenames require it. Until that wiring exists, treat ONNX / ensemble-heavy files as **manual** or **nightly-only**.

## `build.sh` and tests

`build.sh` defaults to **`SKIP_TESTS=1`**, so a plain `./build.sh` does **not** run the full test matrix. For release or pre-merge verification, override explicitly, for example:

```bash
SKIP_TESTS=0 ./build.sh
```

(Exact test steps depend on `build.sh` version; always read the script output.)

## Integration workflow note

`.github/workflows/integration-tests.yml` is **workflow_dispatch** only and historically assumed sibling checkouts (`NeuronDB/`, `NeuronAgent/`, etc.). For tests that live **only** in this repository, use `build-and-test.yml`, extension CI workflows, and `run_test.py` jobs instead.
