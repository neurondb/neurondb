# Versioning Policy

Versioning policy for the NeuronDB extension in this repository.

---

## Semantic versioning

NeuronDB follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): New functionality, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

## Extension versioning

- **Version:** Defined in `neurondb.control` (e.g. `1.0`)
- **Format:** `MAJOR.MINOR` for the extension; tags may use `vMAJOR.MINOR.PATCH`
- **Upgrades:** SQL migration files `neurondb--FROM--TO.sql`

## Compatibility

| Component | PostgreSQL |
|-----------|------------|
| NeuronDB  | 16, 17, 18 |

## Deprecation

1. **Announcement:** Mark deprecated in release notes and docs
2. **Warning period:** At least 2 minor versions with warnings
3. **Removal:** Next major version

## Tags and images

- **Git tags:** `v<version>` (e.g. `v1.0.0`)
- **Docker (default release):** `neurondb/neurondb-cuda` and `ghcr.io/neurondb/neurondb-cuda` with tags per Postgres major **`pg16`**, **`pg17`**, **`pg18`**, **`pg16-<ver>`** … **`pg18-<ver>`**, plus **`latest`** and bare **`<ver>`** for the **PostgreSQL 17** line only (see [container images](../deployment/container-images.md))

## Package versions

- **Python SDK** (if in this repo): version in `setup.py` / `pyproject.toml`, published to PyPI as `neurondb`

---

[Release](README.md) · [Documentation](../readme.md)
