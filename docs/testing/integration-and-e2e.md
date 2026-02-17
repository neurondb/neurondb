# Integration and End-to-End Testing

## Scope

- **NeuronDB → NeuronMCP → NeuronAgent → NeuronDesktop**: Test full flow: create data in NeuronDB, query via MCP tools, run an agent that uses those tools, and verify results in the Desktop UI or API.
- **Model training → deployment → prediction**: Train a model via `neurondb.train()`, deploy it, run predictions via API or MCP, assert on outputs.
- **Agent workflow with tools**: Execute a workflow that invokes SQL/MCP tools and LLM; assert on final state and outputs.

## Running integration tests

- Use the existing `integration-tests.yml` and `integration-tests-full-ecosystem.yml` workflows: they start the stack with Docker Compose and run smoke checks (extension load, version, basic queries).
- For deeper E2E tests, add scripts under `tests/e2e/` (or `NeuronDesktop/tests/e2e/`) that drive the API and optionally the UI (e.g. Playwright) and assert on responses and DB state.

## Chaos testing

- **Kill services**: Stop one container (e.g. NeuronAgent) and verify clients get connection errors and retries; restart and verify recovery.
- **Network partition**: Use Docker network disconnect to isolate a service; verify timeouts and reconnection after partition heals.
- **Disk full**: Fill disk on a node (or use a small volume) and verify graceful failure and logging.

Document scenarios and expected behavior in `docs/operations/chaos-testing.md`.

## Backup and restore verification

- After a full backup (pg_dump or your backup tool), restore to a new instance and run the SQL test suite and integration smoke tests to verify data and extension state.
- Run this periodically (e.g. in CI on a schedule or as a manual release gate).
