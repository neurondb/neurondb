# Release checklist (NeuronDB extension)

Use before tagging or publishing images.

1. **Branch hygiene:** `main` (or release branch) is green on PR workflows: Extension CI, SQL subset, Python SDK, TypeScript SDK, Static Analysis (as applicable).
2. **Extension:** From a clean tree, `make && sudo make install && sudo -u postgres make installcheck`.
3. **SQL corpus:** Run `src/tests/run_test.py` for the categories you ship (see [testing-runbook.md](testing-runbook.md)); set `NDB_CI_SKIP_LIVE=0` only when live embedding tests are intended and credentials exist.
4. **Version files:** Bump `default_version` in [neurondb.control](../../neurondb.control) and add upgrade scripts under `sql/` as required by [versioning-policy.md](../release/versioning-policy.md).
5. **Python SDK:** `cd src/sdks/python && pip install -e ".[dev]" && pytest -v tests/`.
6. **TypeScript SDK:** `cd src/sdks/typescript && npm ci && npm run build && npm test`.
7. **Docker (optional):** Build `docker/neurondb/Dockerfile` locally or rely on [neurondb-docker.yml](../../.github/workflows/neurondb-docker.yml).
8. **Notes:** Summarize breaking SQL/C API changes and link [deprecation-policy.md](deprecation-policy.md) if deprecations are included.
