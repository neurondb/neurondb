# Installation Guide

NeuronDB is a PostgreSQL extension that must be compiled and installed on your PostgreSQL server. This guide walks you through the complete installation process on different platforms.

## Prerequisites

### System Requirements

- **PostgreSQL**: Version 16, 17, or 18
- **Operating System**: Linux (Ubuntu, Debian, Rocky Linux) or macOS
- **RAM**: Minimum 2GB, recommended 8GB+
- **Disk Space**: Minimum 500MB for installation

### Build Dependencies

NeuronDB requires these development libraries:

- **Compiler**: GCC 7.0+ or Clang 10.0+
- **libcurl**: For ML model runtime and HTTP operations
- **OpenSSL**: For encryption features
- **zlib**: For compression

## Installation by Platform

### Ubuntu / Debian

#### Step 1: Add PostgreSQL Repository

```bash
# Import PostgreSQL signing key
sudo apt-get install -y wget gnupg2 lsb-release
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | \
    gpg --dearmor | sudo tee /usr/share/keyrings/postgresql.gpg > /dev/null

# Add repository
echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] \
http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" | \
sudo tee /etc/apt/sources.list.d/pgdg.list
```

#### Step 2: Install PostgreSQL and Development Tools

```bash
# Update package list
sudo apt-get update

# Install PostgreSQL 17 and development packages
sudo apt-get install -y \
    postgresql-17 \
    postgresql-server-dev-17 \
    postgresql-contrib-17

# Install build dependencies
sudo apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    zlib1g-dev \
    pkg-config
```

#### Step 3: Build NeuronDB

```bash
# Clone the repository
git clone https://github.com/neurondb/NeurondB.git
cd NeurondB

# Build the extension
make PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config

# Verify build succeeded
ls -lh neurondb.so
```

#### Step 4: Install NeuronDB

```bash
# Install to PostgreSQL directory
sudo make install PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config

# Verify installation
ls -lh /usr/lib/postgresql/17/lib/neurondb.so
ls -lh /usr/share/postgresql/17/extension/neurondb*
```

### macOS

#### Step 1: Install PostgreSQL via Homebrew

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install PostgreSQL
brew install postgresql@17

# Start PostgreSQL service
brew services start postgresql@17
```

#### Step 2: Build NeuronDB

```bash
# Clone repository
git clone https://github.com/neurondb/NeurondB.git
cd NeurondB

# Build extension (Homebrew PostgreSQL path)
make PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config

# Verify build
ls -lh neurondb.dylib
```

#### Step 3: Install NeuronDB

```bash
# Install extension
sudo make install PG_CONFIG=/opt/homebrew/opt/postgresql@17/bin/pg_config

# Verify installation
ls -lh /opt/homebrew/opt/postgresql@17/lib/neurondb.dylib
```

### Rocky Linux / RHEL

#### Step 1: Add PostgreSQL Repository

```bash
# Install repository RPM
sudo dnf install -y \
    https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-$(uname -m)/pgdg-redhat-repo-latest.noarch.rpm

# Disable built-in PostgreSQL module
sudo dnf -qy module disable postgresql

# Enable PowerTools/CRB repository
sudo dnf config-manager --set-enabled crb
```

#### Step 2: Install PostgreSQL and Development Tools

```bash
# Install PostgreSQL
sudo dnf install -y \
    postgresql17-server \
    postgresql17-devel \
    postgresql17-contrib

# Install build dependencies
sudo dnf install -y \
    gcc \
    make \
    libcurl-devel \
    openssl-devel \
    zlib-devel \
    pkg-config
```

#### Step 3: Initialize PostgreSQL Cluster

```bash
# Initialize database cluster
sudo /usr/pgsql-17/bin/postgresql-17-setup initdb

# Start and enable PostgreSQL
sudo systemctl enable postgresql-17
sudo systemctl start postgresql-17
```

#### Step 4: Build and Install NeuronDB

```bash
# Clone repository
git clone https://github.com/neurondb/NeurondB.git
cd NeurondB

# Build extension
make PG_CONFIG=/usr/pgsql-17/bin/pg_config

# Install extension
sudo make install PG_CONFIG=/usr/pgsql-17/bin/pg_config
```

## Post-Installation Configuration

### Configure Background Workers

NeuronDB includes three background workers that require configuration:

```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/17/main/postgresql.conf

# Add this line to enable NeuronDB workers
shared_preload_libraries = 'neurondb'
```

**Restart PostgreSQL** to apply changes:

```bash
# Ubuntu/Debian
sudo systemctl restart postgresql

# Rocky Linux/RHEL
sudo systemctl restart postgresql-17

# macOS
brew services restart postgresql@17
```

### Create Extension in Database

Connect to your database and create the extension:

```sql
-- Connect to your database
psql -d your_database

-- Create NeuronDB extension
CREATE EXTENSION neurondb;

-- Verify installation
\dx neurondb

-- Check version
SELECT extversion FROM pg_extension WHERE extname = 'neurondb';
```

## Verification

### Test Basic Functionality

```sql
-- Test vector type
SELECT '[1.0, 2.0, 3.0]'::vector;

-- Test distance calculation
SELECT vector_l2_distance('[1.0, 0.0]'::vector, '[0.0, 1.0]'::vector);

-- Test embedding generation
SELECT embed_text('Hello world');

-- Check background workers (if configured)
SELECT * FROM pg_stat_activity WHERE backend_type LIKE '%neuron%';
```

### Run Regression Tests

```bash
cd NeurondB

# Run all regression tests
make installcheck PG_CONFIG=/path/to/pg_config

# View results
cat regression.out
```

Expected output: `All 8 tests passed.`

## Troubleshooting

### Build Errors

**Error: `pg_config: not found`**

Solution: Install PostgreSQL development packages or specify full path to `pg_config`.

**Error: `curl.h: No such file or directory`**

Solution: Install libcurl development package:
```bash
# Ubuntu/Debian
sudo apt-get install libcurl4-openssl-dev

# Rocky/RHEL
sudo dnf install libcurl-devel
```

### Runtime Errors

**Error: `could not load library "neurondb.so"`**

Solution: Verify installation path matches PostgreSQL's library directory:
```sql
SHOW dynamic_library_path;
```

**Error: `extension "neurondb" is not available`**

Solution: Check extension files are in correct location:
```bash
ls -l $(pg_config --sharedir)/extension/neurondb*
```

### Worker Not Starting

If background workers don't start:

1. Verify `shared_preload_libraries` is set correctly
2. Check PostgreSQL logs for error messages
3. Ensure PostgreSQL has been restarted after configuration

```bash
# View PostgreSQL logs
tail -f /var/log/postgresql/postgresql-17-main.log
```

## Upgrading

To upgrade NeuronDB to a newer version:

```bash
# Pull latest changes
cd NeurondB
git pull origin main

# Clean previous build
make clean

# Rebuild and reinstall
make PG_CONFIG=/path/to/pg_config
sudo make install PG_CONFIG=/path/to/pg_config

# Restart PostgreSQL
sudo systemctl restart postgresql

# Update extension in database
psql -d your_database -c "ALTER EXTENSION neurondb UPDATE;"
```

## Uninstallation

To completely remove NeuronDB:

```sql
-- Drop extension from all databases
DROP EXTENSION IF EXISTS neurondb CASCADE;
```

```bash
# Remove extension files
sudo make uninstall PG_CONFIG=/path/to/pg_config

# Remove from shared_preload_libraries
# Edit postgresql.conf and remove 'neurondb' from the list
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with basic operations
- [Configuration](../configuration.md) - Configure NeuronDB for your use case
- [Vector Search](../vector-search/vector-types.md) - Learn core concepts and features

## Learn More

For detailed installation instructions, platform-specific guides, and troubleshooting, visit:

**[Installation Documentation](https://neurondb.ai/docs/installation/)**

