# 🚀 Simple Start Guide

<div align="center">

**Get a working local environment with minimal friction**

[![Quick Start](https://img.shields.io/badge/quick--start-5_min-green)](.)
[![Difficulty](https://img.shields.io/badge/difficulty-easy-brightgreen)](.)
[![Docker](https://img.shields.io/badge/docker-required-blue)](.)

</div>

---

This guide walks you through setup step by step. If you know Docker and PostgreSQL, use the [Technical Quickstart](quickstart.md).

---

## 🎯 Goal

**What you'll accomplish:**
- ✅ Get NeuronDB running locally
- ✅ Create your first vector table
- ✅ Perform your first similarity search
- ✅ Understand the basic concepts

**Time required:** 5-10 minutes

---

## 🛣️ Choose Your Path

**Pick the method that works best for you:**

| Method | Best For | Time | Difficulty | Prerequisites |
|--------|----------|------|------------|---------------|
| **🐳 Docker** (recommended) | Fastest start, easiest setup | 5 min | ⭐ Easy | Docker installed |
| **🔧 Native build** | Custom setup, production-like | 30+ min | ⭐⭐⭐ Advanced | PostgreSQL dev headers, C compiler |

Docker handles all dependencies automatically. You do not need to install PostgreSQL or configure it. Everything runs in isolated containers.

---

## 🐳 Docker Quickstart (Recommended)

### 📋 Prerequisites Checklist

Before you begin, make sure you have:

- [ ] **Docker** 20.10+ installed
- [ ] **Docker Compose** 2.0+ installed
- [ ] **4GB+ RAM** available
- [ ] **5-10 minutes** of time
- [ ] **Ports available**: 5433, 8080, 8081, 3000 (optional)

<details>
<summary><strong>🔍 Verify Docker Installation</strong></summary>

Run these commands to verify Docker is installed correctly:

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker compose version

# Verify Docker is running
docker ps
```

**Expected output:**
```
Docker version 20.10.0 or higher
Docker Compose version v2.0.0 or higher
CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES
```

If you see errors, install Docker from [docker.com](https://www.docker.com/get-started).

</details>

---

### 📝 Step-by-Step Instructions

#### Step 1: Start the Database

**From the repository root**, start NeuronDB:

```bash
# Start just the database (simplest option)
docker compose up -d neurondb

# Or start all services (database + agent + MCP + desktop)
docker compose up -d
```

Docker downloads and starts a PostgreSQL container with the NeuronDB extension. The `-d` flag runs it in the background.

**What to expect:**
- First time: Downloads images (may take 2-5 minutes)
- Subsequent runs: Starts immediately (30-60 seconds)

#### Step 2: Wait for Services to Be Healthy

Check that the service is running:

```bash
# Check service status
docker compose ps
```

**Expected output:**
```
NAME                STATUS
neurondb-cpu        healthy (or running)
```

Wait for "healthy" status. This means PostgreSQL initialized and accepts connections. This takes 30 to 60 seconds.

<details>
<summary><strong>🔍 What if it's not healthy?</strong></summary>

If the service shows as "unhealthy" or keeps restarting:

1. **Check logs:**
   ```bash
   docker compose logs neurondb
   ```

2. **Common issues:**
   - Port 5433 already in use → Stop the conflicting service
   - Not enough memory → Allocate more RAM to Docker
   - Disk space full → Free up disk space

3. **Get help:** See [Troubleshooting Guide](troubleshooting.md)

</details>

#### Step 3: Verify Installation

Connect to the database and verify NeuronDB is installed:

```bash
# Create the extension (if not already created)
docker compose exec neurondb psql -U neurondb -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb;"

# Check the version
docker compose exec neurondb psql -U neurondb -d neurondb -c "SELECT neurondb.version();"
```

**Expected output:**
```
CREATE EXTENSION
 version
---------
 2.0
(1 row)
```

> [!SUCCESS]
> **Great!** If you see version `2.0` (or similar), NeuronDB is installed and working correctly.

#### Step 4: Your First Vector Search

Let's create a simple example to understand how vector search works:

```bash
docker compose exec neurondb psql -U neurondb -d neurondb <<EOF
-- Step 1: Create a table to store documents with embeddings
CREATE TABLE test_vectors (
  id SERIAL PRIMARY KEY,
  name TEXT,
  embedding vector(3)  -- 3-dimensional vectors for this example
);

-- Step 2: Insert some sample data
INSERT INTO test_vectors (name, embedding) VALUES
  ('apple', '[1.0, 0.0, 0.0]'::vector),
  ('banana', '[0.0, 1.0, 0.0]'::vector),
  ('orange', '[0.5, 0.5, 0.0]'::vector);

-- Step 3: Query the data
SELECT name, embedding FROM test_vectors;

-- Step 4: Find the most similar vector to a query
SELECT 
  name,
  embedding <=> '[0.9, 0.1, 0.0]'::vector AS distance
FROM test_vectors
ORDER BY embedding <=> '[0.9, 0.1, 0.0]'::vector
LIMIT 3;

-- Step 5: Clean up
DROP TABLE test_vectors;
EOF
```

**Expected output:**
```
 name   |   embedding    
--------+----------------
 apple  | [1,0,0]
 banana | [0,1,0]
 orange | [0.5,0.5,0]
(3 rows)

 name   |     distance      
--------+-------------------
 apple  | 0.141421356237309
 orange | 0.424264068711929
 banana | 0.905538513813742
(3 rows)
```

> [!NOTE]
> **Understanding the results:** The `<=>` operator calculates cosine distance. Lower distance = more similar. Apple is closest to `[0.9, 0.1, 0.0]` because both are close to `[1, 0, 0]`.

---

## 🔧 Native Quickstart (Advanced)

<details>
<summary><strong>📦 Native Installation Steps</strong></summary>

**For advanced users only** - Requires PostgreSQL development headers and C compiler

### Prerequisites

- PostgreSQL 16, 17, or 18 installed
- PostgreSQL development headers (`postgresql-dev` or `postgresql-devel`)
- C compiler (gcc or clang)
- Make utility

### Installation Steps

#### 1. Build the Extension

```bash
# Navigate to NeuronDB directory
cd NeuronDB

# Build the extension
make

# Install the extension
sudo make install
```

> [!NOTE]
> **What's happening?** The `make` command compiles the C code into a PostgreSQL extension. `make install` copies the compiled files to PostgreSQL's extension directory.

#### 2. Configure PostgreSQL

Add to `postgresql.conf`:

```ini
# Enable the extension
shared_preload_libraries = 'neurondb'
```

Then restart PostgreSQL:

```bash
# On Linux (systemd)
sudo systemctl restart postgresql

# On macOS (Homebrew)
brew services restart postgresql@16
```

#### 3. Create the Extension

Connect to your database:

```bash
psql -d your_database_name
```

Then run:

```sql
-- Create the extension
CREATE EXTENSION neurondb;

-- Verify installation
SELECT neurondb.version();
```

**Expected output:** `2.0`

#### 4. Test with a Basic Query

```sql
-- Create a test table
CREATE TABLE test (
  id SERIAL,
  vec vector(3)
);

-- Insert a vector
INSERT INTO test (vec) VALUES ('[1,2,3]'::vector);

-- Query it
SELECT * FROM test;

-- Clean up
DROP TABLE test;
```

</details>

---

## 🎓 Understanding What You Just Did

<details>
<summary><strong>📚 Key Concepts Explained</strong></summary>

### What is a Vector?

A **vector** is an array of numbers that represents data in a multi-dimensional space. For example:
- `[1.0, 0.0, 0.0]` is a 3-dimensional vector
- In AI, vectors often represent embeddings (dense numerical representations of text, images, etc.)

### What is Vector Search?

**Vector search** finds similar vectors by calculating distances. The `<=>` operator uses cosine distance:
- **Lower distance** = more similar
- **Higher distance** = less similar

### What is an Embedding?

An **embedding** is a vector representation of data (text, images, etc.) that captures semantic meaning. Similar concepts have similar embeddings.

### Why Use NeuronDB?

NeuronDB adds vector search capabilities directly to PostgreSQL, so you can:
- Store vectors alongside your regular data
- Use SQL to query vectors
- Combine vector search with traditional SQL filters
- Leverage PostgreSQL's ACID guarantees

</details>

---

## 🎯 Next Steps

**Continue your journey:**

- [ ] 📐 Read [`architecture.md`](architecture.md) to understand the moving parts
- [ ] 🧪 Try examples from [`examples/`](../../examples/)
- [ ] 📚 Explore the [complete documentation](../../documentation.md)
- [ ] 🔍 If something fails, check [`troubleshooting.md`](troubleshooting.md)
- [ ] 🚀 Try the [Quickstart Data Pack](../../examples/quickstart-data/) for sample data

---

## 💡 Quick Tips

<details>
<summary><strong>💡 Helpful Tips for Success</strong></summary>

### Docker Tips

- **Docker is recommended** for the easiest setup
- **Keep containers running** - They use minimal resources when idle
- **Use `docker compose logs`** to see what's happening
- **Port conflicts?** Change ports in `docker-compose.yml`

### Learning Tips

- **Start simple** - Get it running first, then explore advanced features
- **Read the architecture guide** to understand how components work together
- **Try the examples** - They're designed to teach concepts
- **Check troubleshooting** if you encounter issues

### Development Tips

- **Use the SQL recipes** - Ready-to-run examples in `Docs/getting-started/recipes/`
- **Try the CLI helpers** - Scripts in `scripts/` for common tasks
- **Explore the examples** - Working code in `examples/`

</details>

---

## ❓ Common Questions

<details>
<summary><strong>❓ Frequently Asked Questions</strong></summary>

### Q: Do I need all services running?

**A:** No! You can run just NeuronDB (the database) if you only need vector search. The other services (Agent, MCP, Desktop) are optional.

### Q: Can I use my existing PostgreSQL?

**A:** Yes! You can install NeuronDB into your existing PostgreSQL installation. See [Native Installation](installation-native.md).

### Q: What's the difference between Docker and native install?

**A:** 
- **Docker**: Everything is isolated, easy to remove, no system changes
- **Native**: Direct integration with your PostgreSQL, more control, production-like

### Q: How do I stop everything?

**A:** 
```bash
# Stop all services (keeps data)
docker compose down

# Stop and remove all data
docker compose down -v
```

### Q: Where is my data stored?

**A:** In Docker volumes. Use `docker volume ls` to see them. Data persists even if you stop containers.

</details>

---

## 🔗 Related Documentation

| Document | Description |
|----------|-------------|
| **[Architecture Guide](architecture.md)** | Understand how components work |
| **[Installation Guide](installation.md)** | Detailed installation options |
| **[Troubleshooting](troubleshooting.md)** | Common issues and solutions |
| **[Complete Documentation](../../documentation.md)** | Full documentation index |

---

<div align="center">

[⬆ Back to Top](#-simple-start-guide) · [📚 Main Documentation](../../documentation.md) · [🚀 Quickstart](quickstart.md)

</div>
