# NeuronDB Development Guide

Contribute to the NeuronDB PostgreSQL extension: code layout, adding SQL functions and ML algorithms, and testing.

---

## Code organization

Repository root (no top-level `NeuronDB/` directory):

```
├── src/              # C source code
│   ├── core/         # Core vector operations
│   ├── ml/           # ML algorithms
│   ├── gpu/          # GPU acceleration
│   ├── index/        # Index methods
│   └── tests/        # Test runner and SQL tests
├── sql/              # SQL extension files (neurondb--*.sql)
├── docs/             # Documentation
└── Makefile          # Build (./build.sh or make)
```

---

## Adding new SQL functions

### 1. Implement C function

**File:** `src/vector/vector_ops.c` (or appropriate module)

```c
PG_FUNCTION_INFO_V1(my_new_function);

Datum
my_new_function(PG_FUNCTION_ARGS)
{
    Vector *vec = PG_GETARG_VECTOR_P(0);
    // Implementation
    PG_RETURN_VECTOR_P(result);
}
```

### 2. Add SQL declaration

**File:** `sql/neurondb--1.0.sql` (or the correct versioned SQL file)

```sql
CREATE FUNCTION my_new_function(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'my_new_function'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION my_new_function IS 'Description of function';
```

### 3. Add tests

**File:** `src/tests/sql/basic/my_new_function.sql`

```sql
SELECT my_new_function('[1.0, 2.0, 3.0]'::vector);
```

---

## Adding new ML algorithms

### 1. Implement algorithm

**File:** `src/ml/ml_my_algorithm.c`

### 2. Add SQL functions

**File:** `sql/neurondb--1.0.sql` — `CREATE FUNCTION` for train and predict.

### 3. Register in catalog

**File:** `src/ml/ml_catalog.c` — Register the algorithm in the catalog.

---

## Testing

**SQL regression tests:**

```bash
# From repository root
make installcheck
```

**Add a test:** Create or extend a `.sql` file under `src/tests/sql/` and run `make installcheck`.

**Docker:** See [Testing with Docker](../readme-docker.md).

---

## Debugging

- **PostgreSQL:** `SET client_min_messages = debug1;` and inspect `pg_extension` for `neurondb`.
- **GPU:** `SELECT * FROM neurondb_gpu_info();` and enable debug logging as needed.

---

## Related documentation

- [Build system](build-system.md)
- [Contributing](../../CONTRIBUTING.md)
