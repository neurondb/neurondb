# NeuronDB Model Catalog Demo

This demo exercises the model catalog tables and lifecycle functions introduced in NeuronDB 1.0. It walks through registering models, loading them from disk, querying catalog metadata, auditing events, validating error handling, and cleaning up resources.

## Contents

- `sql/000_run_all.sql` – orchestrates the full walkthrough.
- `sql/001_model_registration.sql` – registers models/versions and shows catalog queries.
- `sql/002_model_lifecycle.sql` – simulates load, predict, fine-tune, export, and logs model events.
- `sql/003_model_audit.sql` – reports catalog status, recent events, and consistency checks.
- `sql/004_model_error_cases.sql` – exercises duplicate/version errors and missing file handling.
- `sql/005_cleanup.sql` – removes demo data and temporary files.

## Requirements

- Run inside `psql` with the NeuronDB extension installed.
- Local filesystem access to create temporary model files (the scripts create dummy ONNX files under `/tmp/neurondb_models_demo`).

## Usage

```bash
cd demo/models
psql -U postgres -d your_db -f sql/000_run_all.sql
```

Each script emits detailed `\echo` output and catalog queries so you can validate model metadata end-to-end.

## Key Catalog Objects

- `neurondb.models` – logical model entries scoped by tenant.
- `neurondb.model_versions` – physical artifacts with storage URIs, format, and status.
- `neurondb.model_events` – audit log of lifecycle events.
- `neurondb.model_catalog` – convenience view joining models to their latest version.

## Functions Used

- `neurondb.ensure_model`
- `neurondb.register_model_version`
- `neurondb.update_model_version_status`
- `neurondb.log_model_event`
- `load_model`, `predict`, `predict_batch`, `finetune_model`, `export_model`

These demos ensure the catalog remains consistent and provide a reference for integrating NeuronDB with higher-level orchestration systems.
