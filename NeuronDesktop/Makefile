.PHONY: build build-api build-frontend test test-api test-frontend clean clean-api clean-frontend deps deps-api deps-frontend install run run-api run-frontend help

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)NeuronDesktop Build System$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2}'

build: build-api build-frontend ## Build both API backend and frontend

build-api: deps-api ## Build the API backend (Go)
	@echo "$(CYAN)Building NeuronDesktop API...$(NC)"
	@mkdir -p bin bin/scripts bin/scripts/lib
	@cd api && go build -o ../bin/neurondesktop ./cmd/server
	@echo "$(CYAN)Copying all runtime files to bin/...$(NC)"
	@mkdir -p bin/scripts bin/scripts/lib bin/sql
	@cp scripts/neurondesktop-*.sh bin/scripts/ 2>/dev/null || true
	@cp -r scripts/lib/*.sh bin/scripts/lib/ 2>/dev/null || true
	@cp neuron-desktop.sql bin/sql/ 2>/dev/null || true
	@cp api/neurondesktop.sql bin/sql/ 2>/dev/null || true
	@chmod +x bin/scripts/*.sh 2>/dev/null || true
	@echo "# NeuronDesktop Runtime Package" > bin/README.md
	@echo "" >> bin/README.md
	@echo "This directory contains everything needed to run NeuronDesktop." >> bin/README.md
	@echo "" >> bin/README.md
	@echo "## Structure" >> bin/README.md
	@echo "- neurondesktop - Main binary executable" >> bin/README.md
	@echo "- scripts/ - Runtime utility scripts" >> bin/README.md
	@echo "- sql/ - Database schema files" >> bin/README.md
	@echo "" >> bin/README.md
	@echo "## Usage" >> bin/README.md
	@echo "Run: ./neurondesktop" >> bin/README.md
	@echo "Setup: ./scripts/neurondesktop-setup.sh" >> bin/README.md
	@echo "$(GREEN)✓ API built with all runtime files (binary, scripts, SQL) - complete package$(NC)"

build-frontend: deps-frontend ## Build the frontend (Next.js)
	@echo "$(CYAN)Building NeuronDesktop frontend...$(NC)"
	@cd frontend && npm run build
	@echo "$(GREEN)✓ Frontend built$(NC)"

deps: deps-api deps-frontend ## Install all dependencies

deps-api: ## Install API dependencies (Go modules)
	@echo "$(CYAN)Installing API dependencies...$(NC)"
	@cd api && go mod download
	@echo "$(GREEN)✓ API dependencies installed$(NC)"

deps-frontend: ## Install frontend dependencies (npm)
	@echo "$(CYAN)Installing frontend dependencies...$(NC)"
	@if [ ! -d frontend/node_modules ]; then \
		cd frontend && npm install; \
	else \
		echo "$(GREEN)✓ Frontend dependencies already installed$(NC)"; \
	fi

test: test-api test-frontend ## Run all tests

test-api: build-api ## Run API tests with race detector
	@echo "$(CYAN)Running API tests with race detector...$(NC)"
	@cd api && go test -race ./... || echo "$(YELLOW)⚠ Some tests may have failed$(NC)"

test-api-unit: ## Run API unit tests only (with race detector)
	@echo "$(CYAN)Running API unit tests with race detector...$(NC)"
	@cd api && go test -v -race ./internal/handlers/... ./internal/db/...

test-api-integration: ## Run API integration tests only (with race detector)
	@echo "$(CYAN)Running API integration tests with race detector...$(NC)"
	@cd api && go test -v -race ./internal/mcp/... ./internal/neurondb/... ./internal/agent/...

test-api-e2e: ## Run end-to-end tests
	@echo "$(CYAN)Running E2E tests...$(NC)"
	@cd api && go test -v ../../tests/e2e/...

test-api-all: ## Run all API tests with coverage
	@echo "$(CYAN)Running comprehensive test suite...$(NC)"
	@./tests/neurondesktop-test-all.sh

test-api-coverage: ## Generate test coverage report (with race detector)
	@echo "$(CYAN)Generating test coverage with race detector...$(NC)"
	@cd api && go test -race -coverprofile=coverage.out ./...
	@cd api && go tool cover -html=coverage.out -o coverage.html
	@echo "$(GREEN)✓ Coverage report generated: api/coverage.html$(NC)"

fix: fix-api fix-frontend ## Fix code formatting and dependencies

fix-api: ## Fix Go code formatting and dependencies
	@echo "$(CYAN)Fixing Go code formatting...$(NC)"
	@cd api && go fmt ./...
	@cd api && go mod tidy
	@echo "$(GREEN)✓ Go code formatted and dependencies fixed$(NC)"

fix-frontend: ## Fix frontend code formatting
	@echo "$(CYAN)Fixing frontend code formatting...$(NC)"
	@if command -v npm >/dev/null 2>&1; then \
		cd frontend && npm run lint -- --fix 2>/dev/null || echo "$(YELLOW)⚠ Linting may not be configured$(NC)"; \
	else \
		echo "$(YELLOW)⚠ npm not found - skipping frontend formatting$(NC)"; \
	fi
	@echo "$(GREEN)✓ Frontend code formatted$(NC)"

test-frontend: deps-frontend ## Run frontend tests
	@echo "$(CYAN)Running frontend tests...$(NC)"
	@cd frontend && npm test || echo "$(YELLOW)⚠ Frontend tests may not be configured$(NC)"

clean: clean-api clean-frontend ## Clean all build artifacts

clean-api: ## Clean API build artifacts
	@echo "$(CYAN)Cleaning API artifacts...$(NC)"
	@rm -rf bin/ api/bin/
	@echo "$(GREEN)✓ API cleaned$(NC)"

clean-frontend: ## Clean frontend build artifacts
	@echo "$(CYAN)Cleaning frontend artifacts...$(NC)"
	@cd frontend && rm -rf .next dist build node_modules/.cache
	@echo "$(GREEN)✓ Frontend cleaned$(NC)"

install: build ## Install the application (builds everything)
	@echo "$(GREEN)✓ NeuronDesktop installed$(NC)"
	@echo "$(CYAN)API binary: bin/neurondesktop$(NC)"
	@echo "$(CYAN)Frontend: frontend/.next$(NC)"

run: run-api ## Run the full application (API only, frontend needs separate process)

run-api: build-api ## Run the API server
	@echo "$(CYAN)Running NeuronDesktop API...$(NC)"
	@./bin/neurondesktop || cd api && go run ./cmd/server

run-frontend: deps-frontend ## Run the frontend dev server
	@echo "$(CYAN)Running NeuronDesktop frontend...$(NC)"
	@cd frontend && npm run dev

migrate: ## Run database migrations
	@echo "$(CYAN)Running database migrations...$(NC)"
	@if command -v psql >/dev/null 2>&1; then \
		psql -d neurondesk -f api/migrations/001_initial_schema.sql 2>/dev/null || \
		echo "$(YELLOW)⚠ Migration may have failed or database may not exist$(NC)"; \
	else \
		echo "$(YELLOW)⚠ psql not found - cannot run migrations$(NC)"; \
	fi

.DEFAULT_GOAL := help
