# Test Suites for neurondb-cloud and neurondb-hub

This document describes how to add and run test suites for the **neurondb-cloud** and **neurondb-hub** products. Those repositories are expected to live as sibling directories of the main neurondb repo (e.g. `../neurondb-cloud`, `../neurondb-hub`) when deploying with `scripts/deploy-all.sh`.

## neurondb-cloud

### Suggested test layout

- **Unit tests**: In `control-plane/internal/*` next to the code (e.g. `internal/auth/handlers_test.go`, `internal/rbac/enforce_test.go`).
- **Integration tests**: `control-plane/tests/integration/` for API tests against the gateway with a test DB.

### Example unit tests

**Auth (signup/login):**

```go
// internal/auth/handlers_test.go
package auth

import (
	"context"
	"net/http/httptest"
	"testing"
	// use test DB or mocks
)

func TestHandleSignup_ValidRequest(t *testing.T) {
	// Setup pool, secret; POST /api/v1/auth/signup with JSON body; assert 201 and JWT in response.
}

func TestHandleLogin_InvalidPassword(t *testing.T) {
	// Create user, then POST login with wrong password; assert 401.
}
```

**RBAC:**

```go
// internal/rbac/enforce_test.go
package rbac

func TestCheckPermission_AdminAll(t *testing.T) {
	// User with admin:all role gets any permission.
}

func TestCheckPermission_NoAssignment(t *testing.T) {
	// User with no role assignments: document desired behavior (allow vs deny).
}
```

### Running tests

From the neurondb-cloud repo root:

```bash
cd control-plane && go test ./...
```

Use a test database URL for integration tests (e.g. `DATABASE_URL` or `DB_HOST` env).

---

## neurondb-hub

### Suggested test layout

- **Unit tests**: In `backend/internal/*` (e.g. `internal/knowledge/store_test.go`, `internal/agent/handlers_test.go`, `internal/auth/jwt_test.go`).
- **Integration tests**: `backend/tests/integration/` for API tests with Hub DB and optionally a mock NeuronAgent.

### Example unit tests

**JWT:**

```go
// internal/auth/jwt_test.go
package auth

import "testing"

func TestRequireJWTSecret_FailsWhenUnset(t *testing.T) {
	// Ensure RequireJWTSecret() returns error or panics when JWT_SECRET is empty.
}

func TestIssueAndVerify(t *testing.T) {
	// Set JWT_SECRET; issue token; verify and check claims.
}
```

**Knowledge store:**

```go
// internal/knowledge/store_test.go
package knowledge

import "testing"

func TestSetContent_GetByID(t *testing.T) {
	// Use in-memory or test DB; create source, SetContent, get by ID and assert content.
}
```

**Agent creation (error handling):**

```go
// internal/agent/handlers_test.go
package agent

import "testing"

func TestCreateAgent_NeuronAgentFails_Returns502(t *testing.T) {
	// Mock NeuronAgent HTTP to return 500; call Hub create agent; assert 502 and no Hub row created.
}
```

### Running tests

From the neurondb-hub repo root:

```bash
cd backend && go test ./...
```

For integration tests, set `DATABASE_URL` (or equivalent) to a test database.

---

## Core neurondb (this repo)

The main neurondb repo already includes:

- **NeuronAgent**: Unit tests for agent (e.g. `internal/agent/modular_rag_test.go`), validation, and other packages.
- **NeuronMCP**: Tests under `internal/`, `pkg/`.
- **NeuronDesktop**: API and integration tests under `api/` and `tests/`.

Run all tests:

```bash
cd NeuronAgent && go test ./...
cd NeuronMCP   && go test ./...
cd NeuronDesktop && go test ./...
```

---

## CI

When Cloud and Hub repos are present, add a CI step (e.g. in GitHub Actions) to run their test suites, for example:

```yaml
- name: Test neurondb-cloud
  run: |
    if [ -d ../neurondb-cloud ]; then
      cd ../neurondb-cloud/control-plane && go test ./...
    fi

- name: Test neurondb-hub
  run: |
    if [ -d ../neurondb-hub ]; then
      cd ../neurondb-hub/backend && go test ./...
    fi
```
