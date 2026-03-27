# Directory structure

Layout of the NeuronDB repository.

## Repository root

```
├── src/                 # C source and tests
│   ├── core/            # Core extension logic
│   ├── vector/          # Vector types and operations
│   ├── ml/              # ML algorithms
│   ├── gpu/              # GPU backends (CUDA, ROCm, Metal)
│   ├── index/           # Index access methods
│   ├── llm/              # LLM and embedding integration
│   └── tests/           # Test runner and SQL tests
├── sql/                 # Extension SQL (neurondb--*.sql)
├── docker/              # Dockerfiles and Compose
├── scripts/             # Build, deploy, and utility scripts
├── docs/                # Documentation
├── Makefile             # Build (includes Makefile.core)
├── build.sh             # Automated build and install
└── INSTALL.md           # Install instructions
```

## Key paths

| Path | Description |
|------|-------------|
| `src/` | Extension C source |
| `sql/` | CREATE EXTENSION SQL and upgrades |
| `docker/docker-compose.yml` | Compose file for NeuronDB (and optional services) |
| `scripts/` | Helper scripts (e.g. quickstart data, verification) |

---

[Documentation](readme.md)
