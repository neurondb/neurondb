# NeuronDB Development Guide

**Complete development guide for contributing to NeuronDB ecosystem.**

> **Version:** 1.0  
> **Last Updated:** 2025-01-01

## Table of Contents

- [Code Organization](#code-organization)
- [Adding New SQL Functions](#adding-new-sql-functions)
- [Adding New ML Algorithms](#adding-new-ml-algorithms)
- [Adding New Tools](#adding-new-tools)
- [Testing Procedures](#testing-procedures)
- [Debugging Guides](#debugging-guides)

---

## Code Organization

### NeuronDB Extension

**Structure:**
```
NeuronDB/
├── src/              # C source code
│   ├── core/         # Core vector operations
│   ├── ml/           # ML algorithms
│   ├── gpu/          # GPU acceleration
│   ├── index/        # Index methods
│   └── ...
├── include/          # Header files
├── tests/            # Test files
└── docs/             # Documentation
```

### NeuronAgent

**Location:** **neuron-agent** repo. Structure (reference):

```
neuron-agent/
├── internal/
│   ├── agent/        # Agent runtime
│   ├── api/          # REST API
│   ├── tools/        # Tool system
│   └── ...
└── cmd/              # Command-line tools
```

### NeuronMCP

**Location:** **neuron-mcp** repo. Structure (reference):

```
neuron-mcp/
├── internal/
│   ├── tools/        # MCP tools
│   ├── server/       # MCP server
│   └── ...
└── cmd/              # Command-line tools
```

---

## Adding New SQL Functions

### Step 1: Implement C Function

**File:** `NeuronDB/src/vector/vector_ops.c`

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

### Step 2: Add SQL Declaration

**File:** `NeuronDB/sql/neurondb--1.0.sql`

```sql
CREATE FUNCTION my_new_function(vector) RETURNS vector
    AS 'MODULE_PATHNAME', 'my_new_function'
    LANGUAGE C IMMUTABLE STRICT;
COMMENT ON FUNCTION my_new_function IS 'Description of function';
```

### Step 3: Add Tests

**File:** `NeuronDB/tests/sql/basic/my_new_function.sql`

```sql
SELECT my_new_function('[1.0, 2.0, 3.0]'::vector);
```

---

## Adding New ML Algorithms

### Step 1: Implement Algorithm

**File:** `NeuronDB/src/ml/ml_my_algorithm.c`

```c
PG_FUNCTION_INFO_V1(train_my_algorithm);

Datum
train_my_algorithm(PG_FUNCTION_ARGS)
{
    // Training implementation
    // Store model in catalog
    PG_RETURN_INT32(model_id);
}
```

### Step 2: Add SQL Functions

**File:** `NeuronDB/sql/neurondb--1.0.sql`

```sql
CREATE FUNCTION train_my_algorithm(text, text, text) RETURNS integer
    AS 'MODULE_PATHNAME', 'train_my_algorithm'
    LANGUAGE C STABLE;

CREATE FUNCTION predict_my_algorithm(integer, vector) RETURNS float8
    AS 'MODULE_PATHNAME', 'predict_my_algorithm'
    LANGUAGE C STABLE STRICT;
```

### Step 3: Register in Catalog

**File:** `NeuronDB/src/ml/ml_catalog.c`

Add algorithm to catalog system.

---

## Adding New Tools

### NeuronMCP Tool

**File:** In **neuron-mcp** repo: `internal/tools/my_tool.go`

```go
type MyTool struct {
    *BaseTool
    executor *QueryExecutor
    logger   *logging.Logger
}

func NewMyTool(db *database.Database, logger *logging.Logger) *MyTool {
    return &MyTool{
        BaseTool: NewBaseTool(
            "my_tool",
            "Tool description",
            InputSchema(),
        ),
        executor: NewQueryExecutor(db),
        logger:   logger,
    }
}

func (t *MyTool) Execute(ctx context.Context, params map[string]interface{}) (*ToolResult, error) {
    // Implementation
    return Success(result, metadata), nil
}
```

**Register:**
```go
// In register.go
registry.Register(NewMyTool(db, logger))
```

---

## Testing Procedures

### SQL Tests

**Run:**
```bash
cd NeuronDB
make installcheck
```

**Add Test:**
```sql
-- File: tests/sql/basic/my_test.sql
BEGIN;
SELECT my_function('[1.0, 2.0, 3.0]'::vector);
ROLLBACK;
```

### Go Tests

**Run:**
```bash
cd NeuronAgent
go test ./...
```

**Add Test:**
```go
func TestMyFunction(t *testing.T) {
    // Test implementation
}
```

---

## Debugging Guides

### PostgreSQL Extension

**Enable Debug Logging:**
```sql
SET client_min_messages = debug1;
```

**Check Extension:**
```sql
SELECT * FROM pg_extension WHERE extname = 'neurondb';
```

### GPU Debugging

**Check GPU Status:**
```sql
SELECT * FROM neurondb_gpu_info();
```

**Enable GPU Logging:**
```sql
SET log_min_messages = debug1;
```

---

## Related Documentation

- [Build System](build-system.md)
- [Contributing Guide](../../CONTRIBUTING.md)
- [Code Standards](../../CONTRIBUTING.md#code-standards)

---

**Last Updated:** 2025-01-01  
**Documentation Version:** 1.0.0



