# NeuronDB Build System

Build and test the NeuronDB PostgreSQL extension.

---

## Makefile structure

**Location:** Repository root `Makefile` (includes `Makefile.core`). There is no top-level `NeuronDB/` directory.

**Main targets:**

| Target | Description |
|--------|-------------|
| `all` | Build the extension |
| `install` | Install into PostgreSQL |
| `clean` | Remove build artifacts |
| `installcheck` | Run regression tests |
| `installcheck-tap` | Run TAP tests |
| `installcheck-gpu` | Run GPU tests (when built with GPU) |

---

## Building the extension

From the repository root:

```bash
# Automated (recommended)
./build.sh

# Manual
PG_CONFIG=/path/to/pg_config make
sudo make install
```

Use `PG_CONFIG` from the PostgreSQL installation you target (e.g. `/usr/bin/pg_config` or Homebrew).

---

## Platform-specific builds

### macOS

- Xcode Command Line Tools, PostgreSQL development headers
- `make PG_CONFIG=/path/to/pg_config`

### Linux

- gcc or clang, make, PostgreSQL development headers
- `make PG_CONFIG=/usr/pgsql-17/bin/pg_config` (or your path)

### GPU backends

| Backend | Build |
|---------|--------|
| **CUDA** | CUDA Toolkit 12.2+; build with CUDA enabled (see `build.sh` / Makefile) |
| **ROCm** | ROCm 5.7+; build with ROCm enabled |
| **Metal** | Apple Silicon, macOS 13+; build with Metal enabled |

See [INSTALL.md](../../INSTALL.md) and [GPU docs](../gpu/gpu-feature-matrix.md).

---

## Dependencies

- **PostgreSQL:** 16, 17, or 18 (headers and `pg_config`)
- **ONNX Runtime:** Optional (for some embedding/LLM features)
- **ML libs:** XGBoost, LightGBM, CatBoost optional; see [INSTALL.md](../../INSTALL.md)

---

## Testing

**SQL regression tests:**

```bash
make installcheck
```

**Location of tests:** `src/tests/sql/` (and TAP in `src/tests/`).

**With Docker:** See [Testing with Docker](readme-docker.md).

---

## Related documentation

- [Development guide](development-guide.md)
- [INSTALL.md](../../INSTALL.md)
- [Contributing](../../CONTRIBUTING.md)
