# NeuronDB as a PostgreSQL Extension

This document describes how NeuronDB is packaged and managed as a **PostgreSQL extension**, following the [official extension packaging rules](https://www.postgresql.org/docs/current/extend-extensions.html).

## Control file

- **Path**: `NeuronDB/neurondb.control`
- **Installed as**: `SHAREDIR/extension/neurondb.control` (e.g. `/usr/share/postgresql/17/extension/`)

Parameters:

| Parameter | Value | Meaning |
|-----------|--------|--------|
| `default_version` | `3.1.0` | Version installed by `CREATE EXTENSION neurondb` when no version is specified |
| `comment` | NeuronDB description | Shown in `pg_extension` and `\dx` |
| `module_pathname` | `$libdir/neurondb` | Shared library name (`.so` or `.dylib`) |
| `relocatable` | `true` | Extension objects can be moved with `ALTER EXTENSION ... SET SCHEMA` |
| `superuser` | `true` | Only superusers can create or update the extension |
| `trusted` | `false` | Installation/update scripts run as superuser; not installable by non-superusers with only `CREATE` on database |

Version identifiers in script names must not contain `--` or leading/trailing `-`. The version `3.0.0-devel` is valid.

## File layout (source vs installed)

**Source (repository):**

```
NeuronDB/
├── neurondb.control
├── Makefile, Makefile.core, ...
├── sql/
│   ├── neurondb--1.0.sql
│   ├── neurondb--2.0.sql
│   ├── neurondb--2.1.0.sql
│   ├── neurondb--3.0.0-devel.sql
│   ├── neurondb--3.1.0.sql
│   ├── neurondb--1.0--2.0.sql
│   ├── neurondb--2.0--2.1.0.sql
│   ├── neurondb--2.1.0--3.0.0-devel.sql
│   ├── neurondb--3.0.0-devel--3.1.0.sql
│   └── ...
├── src/   (C sources → neurondb.so / neurondb.dylib)
└── ...
```

**After `make install`:**

- `neurondb.control` → `SHAREDIR/extension/neurondb.control`
- `sql/neurondb--*.sql` → `SHAREDIR/extension/neurondb--*.sql`
- `neurondb.so` or `neurondb.dylib` → `LIBDIR/neurondb.so` or `neurondb.dylib`

`SHAREDIR` and `LIBDIR` come from `pg_config --sharedir` and `pg_config --pkglibdir`.

## Creating the extension

1. **Install files** (requires superuser or root for copy into PostgreSQL dirs):

   ```bash
   make install PG_CONFIG=/path/to/pg_config
   ```

2. **Configure PostgreSQL** (optional but recommended for full functionality):

   Add to `postgresql.conf` (or use `ALTER SYSTEM` then reload):

   ```conf
   shared_preload_libraries = 'neurondb'
   ```

   Then restart the server. This is **required** for:

   - Background workers: neuranq, neuranmon, neurandefrag, neuranllm
   - Shared memory and worker lifecycle

   Without `shared_preload_libraries`, the extension still loads and vector/ML features work, but background workers will not run.

3. **Create the extension** (in each database that should use NeuronDB):

   ```sql
   CREATE EXTENSION neurondb;
   -- Or into a specific schema:
   CREATE EXTENSION neurondb SCHEMA neurondb;
   ```

4. **Verify**:

   ```sql
   SELECT extname, extversion FROM pg_extension WHERE extname = 'neurondb';
   SELECT neurondb.version();
   ```

## Updating the extension

To upgrade an existing installation to a newer version:

```sql
-- Show available update paths
SELECT * FROM pg_extension_update_paths('neurondb');

-- Upgrade to default version
ALTER EXTENSION neurondb UPDATE;

-- Or to a specific version (if you have that script)
ALTER EXTENSION neurondb UPDATE TO '3.1.0';
```

Update scripts follow the naming convention `neurondb--oldver--newver.sql`. PostgreSQL applies the shortest path of update scripts between the installed version and the target version.

## Dropping the extension

```sql
DROP EXTENSION neurondb;
-- Or with CASCADE to drop dependent objects
DROP EXTENSION neurondb CASCADE;
```

All extension-owned objects (types, functions, operators, schema, etc.) are removed. After upgrading the binary and script files, you can `CREATE EXTENSION neurondb` again in the same database if needed.

## Dump and restore

- **pg_dump**: Dumps only `CREATE EXTENSION neurondb;` (and any configuration table data if registered with `pg_extension_config_dump`). It does not dump the extension’s internal object definitions.
- **pg_restore** / **psql -f**: Restoring a dump that includes `CREATE EXTENSION neurondb` requires the same NeuronDB control and script files (and shared library) to be present in the target cluster’s `extension` and `lib` directories; then the extension is recreated from those files.

## Summary

- NeuronDB is a **relocatable**, **superuser-only** extension with versioned SQL scripts and upgrade paths.
- For full functionality (especially background workers), set `shared_preload_libraries = 'neurondb'` and restart.
- Use `CREATE EXTENSION` / `ALTER EXTENSION UPDATE` / `DROP EXTENSION` for lifecycle; use `make install` to deploy files into the PostgreSQL installation.
