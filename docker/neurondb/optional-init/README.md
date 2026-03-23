# Optional init SQL (not run by default)

Scripts here expect SQL assets from sibling repos (NeuronMCP, NeuronAgent). To use them, copy into the image or mount under `/docker-entrypoint-initdb.d/` and rename to `.sql` if needed.

- `neurondb_mcp_schema.sql` — NeuronMCP schema orchestration
- `neurondb_agent_schema.sql` — NeuronAgent migration orchestration

The default CPU/GPU images build from this repository only and create the `neurondb` extension without MCP/Agent extras.
