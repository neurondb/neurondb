# Running NeuronDB Tests with Docker PostgreSQL

This guide explains how to run NeuronDB tests against a Docker PostgreSQL container.
**PostgreSQL and NeuronDB must run in Docker.**

## Quick Start

### Using Docker (Pre-built Container)

The easiest way to run tests with Docker is using the helper script:

```bash
cd /home/pge/pge/neurondb
./scripts/run_tests_docker.sh --compute=cpu
```

**Note:** Docker containers may contain older versions. For testing with the latest source code, use the local PostgreSQL option below.

### Using Local PostgreSQL (Latest Source Code) - **RECOMMENDED**

To test against a local PostgreSQL instance built from the latest source code:

```bash
cd /home/pge/pge/neurondb
./scripts/run_tests_docker.sh --compute=cpu --local --port=5432
```

**This is the recommended method for development and testing**, as:
- ✅ All tests pass with the latest source code
- ✅ You're testing the actual code you're working on
- ✅ No version mismatch issues
- ✅ Automatic PostgreSQL restart works (no Docker limitations)

**Note:** Docker containers may contain outdated code and can have test failures. Always use `--local` when testing your latest changes.

## Docker PostgreSQL Configuration

From `docker-compose.yml`, the default ports are:

- **CPU variant**: Port `5433` (host) → `5432` (container)
- **CUDA variant**: Port `5434` (host) → `5432` (container)
- **ROCm variant**: Port `5435` (host) → `5432` (container)
- **Metal variant**: Port `5436` (host) → `5432` (container)

Default credentials:
- **User**: `neurondb`
- **Password**: `neurondb`
- **Database**: `neurondb`
- **Host**: `localhost` (when connecting from host)

## Using the Helper Script

### Basic Usage

```bash
# Run basic tests with CPU mode (auto-detects container and port)
./scripts/run_tests_docker.sh --compute=cpu

# Run tests on specific port
./scripts/run_tests_docker.sh --compute=cpu --port=5433

# Run all test categories
./scripts/run_tests_docker.sh --compute=cpu --category=all

# Run with verbose output
./scripts/run_tests_docker.sh --compute=cpu --verbose
```

### Available Options

- `--port PORT`: Docker PostgreSQL port (default: auto-detect or 5433)
- `--compute MODE`: Compute mode: `cpu`, `gpu`, `auto` (default: `cpu`)
- `--category CATEGORY`: Test category: `basic`, `advance`, `negative`, `all` (default: `basic`)
- `--container NAME`: Docker container name (default: auto-detect)
- `--user USER`: Database user (default: `neurondb`)
- `--password PASS`: Database password (default: `neurondb`)
- `--db NAME`: Database name (default: `neurondb`)
- `--host HOST`: Database host (default: `localhost`)
- `--verbose`: Enable verbose output
- `--test TEST_NAME`: Run specific test
- `--module MODULE`: Run tests for specific module
- `--help`: Show help message

## Manual Method

You can also run the test script directly (PostgreSQL must be running in Docker):

```bash
cd NeuronDB/tests
python3 run_test.py \
  --compute=cpu \
  --port=5433 \
  --host=localhost \
  --user=neurondb \
  --password=neurondb \
  --db=neurondb \
  --category=basic
```

## Finding the Docker PostgreSQL Port

If you're unsure which port your Docker container is using:

```bash
# Check running containers and their port mappings
docker ps --format "table {{.Names}}\t{{.Ports}}"

# Or check specific container
docker port neurondb-cpu  # For CPU container
docker port neurondb-cuda  # For CUDA container
```

## Docker vs Local PostgreSQL

### When to Use Docker
- Quick testing without building from source
- Testing in isolated environment
- CI/CD pipelines

### When to Use Local PostgreSQL (--local flag) - **RECOMMENDED**
- ✅ **Testing latest source code changes** (all tests pass)
- ✅ **Development workflow** (test your actual code)
- ✅ **Debugging issues** that don't appear in Docker
- ✅ **Docker container has outdated version** (common issue)

**Important:** 
- The source code version runs with **all tests passing**
- Docker containers often have **outdated code** and may show test failures
- **Always use `--local` for development and testing** your latest changes
- Only use Docker for quick smoke tests or when you specifically need containerized testing

## Important Notes

### PostgreSQL Restart Limitation (Docker Only)

The test script attempts to restart PostgreSQL using `pg_ctl` to apply `ALTER SYSTEM` changes for compute mode. **This won't work with Docker containers** because:

1. The script looks for `pg_ctl` and PostgreSQL data directory on the host
2. Docker containers manage their own PostgreSQL instances

**Workaround**: If the script fails with restart errors:

1. **Option 1**: Restart the Docker container manually after the script sets the compute mode:
   ```bash
   docker restart neurondb-cpu
   ```

2. **Option 2**: Pre-configure the container with the correct compute mode before running tests:
   ```bash
   docker exec -it neurondb-cpu psql -U neurondb -d neurondb -c "ALTER SYSTEM SET neurondb.compute_mode = 0;"
   docker restart neurondb-cpu
   ```

3. **Option 3**: If compute mode is already correct, the script will skip the restart automatically.

### Connection Verification

The helper script will verify the connection before running tests. If connection fails:
- Ensure Docker container is running: `docker ps | grep neurondb`
- Verify port is correct: `docker port <container-name>`
- Check credentials match docker-compose.yml settings

## Example Complete Workflow

```bash
# 1. Ensure Docker container is running
docker ps | grep neurondb-cpu

# 2. Verify port (should show 0.0.0.0:5433->5432/tcp)
docker port neurondb-cpu

# 3. Run tests using helper script
./scripts/run_tests_docker.sh --compute=cpu --category=basic

# 4. If restart is needed, restart container manually
docker restart neurondb-cpu
# Then re-run the test command
```

## Troubleshooting

### Connection Failed

```bash
# Check if container is running
docker ps | grep neurondb

# Check container logs
docker logs neurondb-cpu

# Verify port mapping
docker port neurondb-cpu
```

### Restart Failed

If you see errors about PostgreSQL restart:
1. The script will set the compute mode using `ALTER SYSTEM`
2. Manually restart the container: `docker restart neurondb-cpu`
3. Re-run the tests

### Extension Not Found

Ensure the NeuronDB extension is installed in the container:

```bash
docker exec -it neurondb-cpu psql -U neurondb -d neurondb -c "\dx"
```

You should see `neurondb` in the list of extensions.

## Files Involved

- `scripts/run_tests_docker.sh`: Helper script for running tests with Docker
- `NeuronDB/tests/run_test.py`: Main test runner script
- `docker-compose.yml`: Docker configuration with port mappings
- `scripts/neurondb-docker.sh test`: Basic connectivity test script

