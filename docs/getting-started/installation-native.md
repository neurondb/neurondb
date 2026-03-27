# Native Installation Guide

Install the NeuronDB PostgreSQL extension on your system without Docker.

## Prerequisites

### System requirements

- **Operating system:** Linux or macOS
- **PostgreSQL:** 16, 17, or 18
- **Build:** C compiler (GCC or Clang), Make, PostgreSQL server development headers (`pg_config`)

### Verify prerequisites

```bash
pg_config --version
make --version
```

See [INSTALL.md](../../INSTALL.md) for ML library prerequisites (XGBoost, LightGBM, CatBoost) if you need ML features.

## Quick install

From the repository root:

```bash
./build.sh
```

The script installs dependencies where possible, builds the extension, and reports status. For manual steps, see below.

## Manual build and install

### 1. Build the extension

```bash
# From repository root
PG_CONFIG=/path/to/pg_config make
sudo PG_CONFIG=/path/to/pg_config make install
```

Use the `pg_config` from the PostgreSQL installation you want to use (e.g. `/usr/bin/pg_config` or from Homebrew).

### 2. Configure PostgreSQL

Add to `postgresql.conf`:

```ini
shared_preload_libraries = 'neurondb'
```

Then restart PostgreSQL:

```bash
# Linux (systemd)
sudo systemctl restart postgresql

# macOS (Homebrew)
brew services restart postgresql@17
```

### 3. Create the extension

```bash
createdb neurondb
psql -d neurondb -c "CREATE EXTENSION neurondb;"
```

### 4. Verify

```bash
psql -d neurondb -c "SELECT neurondb.version();"
```

## Configuration

Extension configuration is via PostgreSQL GUCs. See [Configuration](../configuration.md). No separate config directory is required.

## Next steps

- [Installation overview](installation.md) — Docker and other options
- [Quick Start](quickstart.md) — Run your first queries

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **pg_config not found** | Install PostgreSQL server dev package (e.g. `postgresql-server-dev-17` on Debian/Ubuntu, or `brew install postgresql` on macOS). |
| **Build errors** | See [INSTALL.md](../../INSTALL.md); run `make clean` then `./build.sh` again. |
| **Extension not found after install** | Ensure `shared_preload_libraries = 'neurondb'` is set and PostgreSQL was restarted. |
