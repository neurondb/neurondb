# -------------------------------------------------------------------------
#
# Makefile
#    Unified build system for NeuronDB Ecosystem
#
# Unified build system supporting both Docker orchestration and source builds.
# Build Modes: Docker (docker-* targets for containerized builds), Source
# (build-* targets for native source builds).
#
# Copyright (c) 2024-2026, neurondb, Inc.
#
# IDENTIFICATION
#    Makefile
#
# -------------------------------------------------------------------------

.PHONY: help \
        docker-build docker-build-cpu docker-build-cuda docker-build-rocm docker-build-metal \
        docker-run docker-run-cpu docker-run-cuda docker-run-rocm docker-run-metal \
        docker-stop docker-logs docker-ps docker-status docker-health docker-clean \
        build build-neurondb build-neuronagent build-neuronmcp build-neurondesktop \
        test test-neurondb test-neuronagent test-neuronmcp test-neurondesktop \
        clean clean-neurondb clean-neuronagent clean-neuronmcp clean-neurondesktop \
        install install-neurondb install-neuronagent install-neuronmcp install-neurondesktop

# Default target
.DEFAULT_GOAL := help

# Docker Compose file and command
COMPOSE_FILE := docker-compose.yml
DOCKER_COMPOSE := $(shell command -v docker-compose >/dev/null 2>&1 && echo docker-compose || echo docker compose)

# Colors for output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║         NeuronDB Ecosystem - Unified Build System           ║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)Docker Build & Run (Containerized)$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(GREEN)Docker Build:$(NC)"
	@echo "  make docker-build          Build all services (CPU variant)"
	@echo "  make docker-build-cpu       Build CPU variant only"
	@echo "  make docker-build-cuda      Build CUDA GPU variant"
	@echo "  make docker-build-rocm      Build ROCm GPU variant"
	@echo "  make docker-build-metal     Build Metal GPU variant"
	@echo ""
	@echo "$(GREEN)Docker Run:$(NC)"
	@echo "  make docker-run             Start all services (CPU)"
	@echo "  make docker-run-cuda        Start all services with CUDA GPU"
	@echo "  make docker-run-rocm         Start all services with ROCm GPU"
	@echo "  make docker-run-metal        Start all services with Metal GPU"
	@echo ""
	@echo "$(GREEN)Docker Management:$(NC)"
	@echo "  make docker-stop            Stop all running services"
	@echo "  make docker-logs            View logs from all services"
	@echo "  make docker-ps              Show running containers"
	@echo "  make docker-status          Show service status"
	@echo "  make docker-health          Check service health"
	@echo "  make docker-clean           Stop and remove containers"
	@echo ""
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo "$(GREEN)Source Build (Native)$(NC)"
	@echo "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(GREEN)Dependency Check:$(NC)"
	@echo "  make check-deps             Check if all dependencies are installed"
	@echo "  make check-deps-neurondb    Check NeuronDB dependencies"
	@echo "  make check-deps-neuronagent Check NeuronAgent dependencies"
	@echo "  make check-deps-neuronmcp   Check NeuronMCP dependencies"
	@echo "  make check-deps-neurondesktop Check NeuronDesktop dependencies"
	@echo ""
	@echo "$(GREEN)Source Build:$(NC)"
	@echo "  make build                  Build all components from source"
	@echo "  make build-neurondb         Build NeuronDB from source (uses build.sh)"
	@echo "  make build-neuronagent      Build NeuronAgent from source"
	@echo "  make build-neuronmcp        Build NeuronMCP from source"
	@echo "  make build-neurondesktop    Build NeuronDesktop from source"
	@echo ""
	@echo "$(GREEN)Debugging & Troubleshooting:$(NC)"
	@echo "  make check-deps-verbose     Check dependencies with detailed output"
	@echo "  make debug-build            Show what would happen during build"
	@echo "  make fix-locks              Remove stale package manager locks"
	@echo "  make build-neurondb-now      Build immediately (skip deps, verbose)"
	@echo ""
	@echo "$(GREEN)Source Test:$(NC)"
	@echo "  make test                   Run tests for all components"
	@echo "  make test-neurondb          Run NeuronDB tests"
	@echo "  make test-neuronagent       Run NeuronAgent tests"
	@echo "  make test-neuronmcp         Run NeuronMCP tests"
	@echo "  make test-neurondesktop     Run NeuronDesktop tests"
	@echo ""
	@echo "$(GREEN)Source Install:$(NC)"
	@echo "  make install                Install all components"
	@echo "  make install-neurondb       Install NeuronDB extension"
	@echo "  make install-neuronagent    Install NeuronAgent"
	@echo "  make install-neuronmcp      Install NeuronMCP"
	@echo "  make install-neurondesktop  Install NeuronDesktop"
	@echo ""
	@echo "$(GREEN)Source Clean:$(NC)"
	@echo "  make clean                  Clean all build artifacts"
	@echo "  make clean-neurondb         Clean NeuronDB artifacts"
	@echo "  make clean-neuronagent      Clean NeuronAgent artifacts"
	@echo "  make clean-neuronmcp        Clean NeuronMCP artifacts"
	@echo "  make clean-neurondesktop    Clean NeuronDesktop artifacts"
	@echo ""
	@echo "$(YELLOW)Note:$(NC) Use 'docker-*' for containerized builds, or 'build-*' for source builds"
	@echo ""

# ============================================================================
# Docker Build Commands
# ============================================================================

docker-build: docker-build-cpu ## Build all services (CPU variant, same as docker-build-cpu)

docker-build-cpu: ## Build CPU variant only
	@echo "$(CYAN)Building NeuronDB (CPU)...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile default build neurondb
	@echo "$(CYAN)Building NeuronAgent...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile default build neuronagent
	@echo "$(CYAN)Building NeuronMCP...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile default build neuronmcp
	@echo "$(CYAN)Building NeuronDesktop API...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile default build neurondesk-api
	@echo "$(CYAN)Building NeuronDesktop Frontend...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile default build neurondesk-frontend
	@echo "$(GREEN)✓ Build complete (CPU)$(NC)"

docker-build-cuda: ## Build CUDA GPU variant
	@echo "$(CYAN)Building NeuronDB (CUDA)...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile cuda build neurondb-cuda
	@echo "$(CYAN)Building NeuronAgent...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile cuda build neuronagent-cuda
	@echo "$(CYAN)Building NeuronMCP...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile cuda build neuronmcp-cuda
	@echo "$(GREEN)✓ Build complete (CUDA)$(NC)"

docker-build-rocm: ## Build ROCm GPU variant
	@echo "$(CYAN)Building NeuronDB (ROCm)...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile rocm build neurondb-rocm
	@echo "$(CYAN)Building NeuronAgent...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile rocm build neuronagent-rocm
	@echo "$(CYAN)Building NeuronMCP...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile rocm build neuronmcp-rocm
	@echo "$(GREEN)✓ Build complete (ROCm)$(NC)"

docker-build-metal: ## Build Metal GPU variant
	@echo "$(CYAN)Building NeuronDB (Metal)...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile metal build neurondb-metal
	@echo "$(CYAN)Building NeuronAgent...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile metal build neuronagent-metal
	@echo "$(CYAN)Building NeuronMCP...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile metal build neuronmcp-metal
	@echo "$(GREEN)✓ Build complete (Metal)$(NC)"

# ============================================================================
# Docker Run Commands
# ============================================================================

docker-run: docker-run-cpu ## Start all services (CPU, same as docker-run-cpu)

docker-run-cpu: ## Start all services (CPU variant)
	@echo "$(CYAN)Starting NeuronDB Ecosystem (CPU)...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. Using defaults.$(NC)"; \
		echo "$(YELLOW)Copy env.example to .env to customize settings.$(NC)"; \
	fi
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile default up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(CYAN)NeuronDB:    localhost:5433$(NC)"
	@echo "$(CYAN)NeuronAgent: http://localhost:8080$(NC)"
	@echo "$(CYAN)NeuronMCP:   Running (stdio protocol)$(NC)"
	@echo ""
	@echo "$(YELLOW)Use 'make docker-logs' to view logs$(NC)"
	@echo "$(YELLOW)Use 'make docker-status' to check service status$(NC)"

docker-run-cuda: ## Start all services with CUDA GPU
	@echo "$(CYAN)Starting NeuronDB Ecosystem (CUDA GPU)...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. Using defaults.$(NC)"; \
		echo "$(YELLOW)Copy env.example to .env and set DB_HOST=neurondb-cuda for GPU support.$(NC)"; \
	fi
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile cuda up -d
	@echo "$(GREEN)✓ Services started (CUDA)$(NC)"
	@echo "$(CYAN)NeuronDB:    localhost:5434$(NC)"
	@echo "$(CYAN)NeuronAgent: http://localhost:8080$(NC)"
	@echo "$(CYAN)NeuronMCP:   Running (stdio protocol)$(NC)"
	@echo ""
	@echo "$(YELLOW)Note: Update .env with DB_HOST=neurondb-cuda and NEURONDB_HOST=neurondb-cuda$(NC)"
	@echo "$(YELLOW)Use 'make docker-logs' to view logs$(NC)"

docker-run-rocm: ## Start all services with ROCm GPU
	@echo "$(CYAN)Starting NeuronDB Ecosystem (ROCm GPU)...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. Using defaults.$(NC)"; \
		echo "$(YELLOW)Copy env.example to .env and set DB_HOST=neurondb-rocm for GPU support.$(NC)"; \
	fi
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile rocm up -d
	@echo "$(GREEN)✓ Services started (ROCm)$(NC)"
	@echo "$(CYAN)NeuronDB:    localhost:5435$(NC)"
	@echo "$(CYAN)NeuronAgent: http://localhost:8080$(NC)"
	@echo "$(CYAN)NeuronMCP:   Running (stdio protocol)$(NC)"
	@echo ""
	@echo "$(YELLOW)Note: Update .env with DB_HOST=neurondb-rocm and NEURONDB_HOST=neurondb-rocm$(NC)"
	@echo "$(YELLOW)Use 'make docker-logs' to view logs$(NC)"

docker-run-metal: ## Start all services with Metal GPU
	@echo "$(CYAN)Starting NeuronDB Ecosystem (Metal GPU)...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Warning: .env file not found. Using defaults.$(NC)"; \
		echo "$(YELLOW)Copy env.example to .env and set DB_HOST=neurondb-metal for GPU support.$(NC)"; \
	fi
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) --profile metal up -d
	@echo "$(GREEN)✓ Services started (Metal)$(NC)"
	@echo "$(CYAN)NeuronDB:    localhost:5436$(NC)"
	@echo "$(CYAN)NeuronAgent: http://localhost:8080$(NC)"
	@echo "$(CYAN)NeuronMCP:   Running (stdio protocol)$(NC)"
	@echo ""
	@echo "$(YELLOW)Note: Update .env with DB_HOST=neurondb-metal and NEURONDB_HOST=neurondb-metal$(NC)"
	@echo "$(YELLOW)Use 'make docker-logs' to view logs$(NC)"

# ============================================================================
# Docker Management Commands
# ============================================================================

docker-stop: ## Stop all running services
	@echo "$(CYAN)Stopping services...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) stop
	@echo "$(GREEN)✓ Services stopped$(NC)"

docker-logs: ## View logs from all services
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps

docker-status: docker-ps ## Show service status (alias for docker-ps)
	@echo ""
	@echo "$(CYAN)Service Health:$(NC)"
	@$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) ps --format json | grep -q '"Health":"healthy"' && echo "$(GREEN)✓ Services healthy$(NC)" || echo "$(YELLOW)⚠ Some services may not be healthy$(NC)"

docker-health: ## Check service health
	@echo "$(CYAN)Checking service health...$(NC)"
	@echo ""
	@echo "$(CYAN)NeuronDB:$(NC)"
	@docker exec neurondb-cpu pg_isready -U neurondb 2>/dev/null && echo "$(GREEN)✓ NeuronDB (CPU) is ready$(NC)" || echo "$(YELLOW)⚠ NeuronDB (CPU) not ready$(NC)"
	@echo ""
	@echo "$(CYAN)NeuronAgent:$(NC)"
	@curl -s http://localhost:8080/health > /dev/null 2>&1 && echo "$(GREEN)✓ NeuronAgent is responding$(NC)" || echo "$(YELLOW)⚠ NeuronAgent not responding$(NC)"
	@echo ""
	@echo "$(CYAN)NeuronMCP:$(NC)"
	@docker exec neurondb-mcp test -f /app/neurondb-mcp -a -x /app/neurondb-mcp 2>/dev/null && echo "$(GREEN)✓ NeuronMCP binary is ready$(NC)" || echo "$(YELLOW)⚠ NeuronMCP binary not found$(NC)"

docker-clean: ## Stop and remove containers
	@echo "$(CYAN)Stopping and removing containers...$(NC)"
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ Containers removed$(NC)"

# ============================================================================
# Dependency Checking
# ============================================================================

check-deps: check-deps-neurondb check-deps-neuronagent check-deps-neuronmcp check-deps-neurondesktop ## Check if dependencies are installed

check-deps-neurondb: ## Check NeuronDB dependencies
	@echo "$(CYAN)Checking NeuronDB dependencies...$(NC)"
	@missing=0; \
	if ! command -v pg_config >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ pg_config not found - PostgreSQL dev headers needed$(NC)"; \
		missing=1; \
	else \
		echo "$(GREEN)✓ pg_config found$(NC)"; \
	fi; \
	if ! command -v make >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ make not found$(NC)"; \
		missing=1; \
	else \
		echo "$(GREEN)✓ make found$(NC)"; \
	fi; \
	if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ C compiler (gcc/clang) not found$(NC)"; \
		missing=1; \
	else \
		echo "$(GREEN)✓ C compiler found$(NC)"; \
	fi; \
	if [ $$missing -eq 0 ]; then \
		echo "$(GREEN)✓ All NeuronDB dependencies satisfied$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Some dependencies missing - build.sh will install them$(NC)"; \
	fi

check-deps-neuronagent: ## Check NeuronAgent dependencies
	@echo "$(CYAN)Checking NeuronAgent dependencies...$(NC)"
	@missing=0; \
	if ! command -v go >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ Go compiler not found (need Go 1.24+)$(NC)"; \
		missing=1; \
	else \
		go_version=$$(go version 2>/dev/null | awk '{print $$3}' | sed 's/go//'); \
		major=$$(echo $$go_version | cut -d. -f1); \
		minor=$$(echo $$go_version | cut -d. -f2); \
		if [ $$major -lt 1 ] || ([ $$major -eq 1 ] && [ $$minor -lt 24 ]); then \
			echo "$(YELLOW)⚠ Go version too old (have $$go_version, need 1.24+)$(NC)"; \
			missing=1; \
		else \
			echo "$(GREEN)✓ Go $$go_version found$(NC)"; \
		fi; \
	fi; \
	if [ $$missing -eq 0 ]; then \
		echo "$(GREEN)✓ All NeuronAgent dependencies satisfied$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Go needs to be installed$(NC)"; \
	fi

check-deps-neuronmcp: ## Check NeuronMCP dependencies
	@echo "$(CYAN)Checking NeuronMCP dependencies...$(NC)"
	@missing=0; \
	if ! command -v go >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ Go compiler not found (need Go 1.24+)$(NC)"; \
		missing=1; \
	else \
		go_version=$$(go version 2>/dev/null | awk '{print $$3}' | sed 's/go//'); \
		major=$$(echo $$go_version | cut -d. -f1); \
		minor=$$(echo $$go_version | cut -d. -f2); \
		if [ $$major -lt 1 ] || ([ $$major -eq 1 ] && [ $$minor -lt 24 ]); then \
			echo "$(YELLOW)⚠ Go version too old (have $$go_version, need 1.24+)$(NC)"; \
			missing=1; \
		else \
			echo "$(GREEN)✓ Go $$go_version found$(NC)"; \
		fi; \
	fi; \
	if [ $$missing -eq 0 ]; then \
		echo "$(GREEN)✓ All NeuronMCP dependencies satisfied$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Go needs to be installed$(NC)"; \
	fi

check-deps-neurondesktop: ## Check NeuronDesktop dependencies
	@echo "$(CYAN)Checking NeuronDesktop dependencies...$(NC)"
	@missing=0; \
	if ! command -v go >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ Go compiler not found (need Go 1.24+)$(NC)"; \
		missing=1; \
	else \
		go_version=$$(go version 2>/dev/null | awk '{print $$3}' | sed 's/go//'); \
		major=$$(echo $$go_version | cut -d. -f1); \
		minor=$$(echo $$go_version | cut -d. -f2); \
		if [ $$major -lt 1 ] || ([ $$major -eq 1 ] && [ $$minor -lt 24 ]); then \
			echo "$(YELLOW)⚠ Go version too old (have $$go_version, need 1.24+)$(NC)"; \
			missing=1; \
		else \
			echo "$(GREEN)✓ Go $$go_version found$(NC)"; \
		fi; \
	fi; \
	if ! command -v npm >/dev/null 2>&1 && ! command -v node >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ Node.js/npm not found (need Node.js 18+)$(NC)"; \
		missing=1; \
	else \
		if command -v node >/dev/null 2>&1; then \
			node_version=$$(node --version 2>/dev/null | sed 's/v//'); \
			echo "$(GREEN)✓ Node.js $$node_version found$(NC)"; \
		fi; \
		if command -v npm >/dev/null 2>&1; then \
			npm_version=$$(npm --version 2>/dev/null); \
			echo "$(GREEN)✓ npm $$npm_version found$(NC)"; \
		fi; \
	fi; \
	if [ $$missing -eq 0 ]; then \
		echo "$(GREEN)✓ All NeuronDesktop dependencies satisfied$(NC)"; \
	else \
		echo "$(YELLOW)⚠ Some dependencies need to be installed$(NC)"; \
	fi

check-deps-verbose: ## Check dependencies with detailed output and package manager status
	@echo "$(CYAN)=== Detailed Dependency Check ===$(NC)"
	@echo ""
	@echo "$(GREEN)System Information:$(NC)"
	@uname -a 2>/dev/null || echo "  (uname not available)"
	@echo ""
	@echo "$(GREEN)Package Manager Status:$(NC)"
	@if command -v apt-get >/dev/null 2>&1; then \
		if pgrep -f "(apt|dpkg)" >/dev/null 2>&1 || [ -f /var/lib/dpkg/lock-frontend ] || [ -f /var/lib/dpkg/lock ]; then \
			echo "$(YELLOW)⚠ apt/dpkg is locked or running$(NC)"; \
			ps aux | grep -E "(apt|dpkg)" | grep -v grep | head -3 || true; \
			echo "$(YELLOW)  Lock files:$(NC)"; \
			ls -la /var/lib/dpkg/lock* 2>/dev/null || echo "    (none)"; \
		else \
			echo "$(GREEN)✓ apt/dpkg is available$(NC)"; \
		fi; \
	elif command -v dnf >/dev/null 2>&1; then \
		if pgrep -f "(dnf|yum|rpm)" >/dev/null 2>&1 || [ -f /var/run/dnf.pid ]; then \
			echo "$(YELLOW)⚠ dnf/yum is locked or running$(NC)"; \
			ps aux | grep -E "(dnf|yum|rpm)" | grep -v grep | head -3 || true; \
		else \
			echo "$(GREEN)✓ dnf/yum is available$(NC)"; \
		fi; \
	elif command -v brew >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Homebrew is available$(NC)"; \
	else \
		echo "$(YELLOW)⚠ No recognized package manager found$(NC)"; \
	fi
	@echo ""
	@$(MAKE) check-deps
	@echo ""
	@echo "$(GREEN)Recommendation:$(NC)"
	@if [ -f /var/lib/dpkg/lock-frontend ] || [ -f /var/lib/dpkg/lock ]; then \
		echo "$(YELLOW)⚠ Package manager lock files detected$(NC)"; \
		echo "$(YELLOW)  If build is stuck, try:$(NC)"; \
		echo "$(YELLOW)  sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock$(NC)"; \
	fi

debug-build: check-deps-neurondb ## Show what would happen during build (dry-run)
	@echo "$(CYAN)=== Build Simulation ===$(NC)"
	@echo ""
	@needs_install=0; \
	if ! command -v pg_config >/dev/null 2>&1; then needs_install=1; fi; \
	if ! command -v make >/dev/null 2>&1; then needs_install=1; fi; \
	if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then needs_install=1; fi; \
	if [ $$needs_install -eq 0 ]; then \
		echo "$(GREEN)Would run:$(NC)"; \
		echo "  cd NeuronDB && SKIP_DEP_INSTALL=1 VERBOSE=1 ./build.sh"; \
		echo ""; \
		echo "$(GREEN)This will:$(NC)"; \
		echo "  ✓ Skip dependency installation"; \
		echo "  ✓ Skip waiting for package manager"; \
		echo "  ✓ Go straight to build"; \
	else \
		echo "$(YELLOW)Would run:$(NC)"; \
		echo "  cd NeuronDB && VERBOSE=1 ./build.sh"; \
		echo ""; \
		echo "$(YELLOW)This will:$(NC)"; \
		echo "  ⚠ Install missing dependencies"; \
		echo "  ⚠ Wait for package manager if locked"; \
		echo "  ✓ Then build"; \
	fi

fix-locks: ## Check and provide instructions to remove stale package manager locks
	@echo "$(CYAN)Checking for package manager locks...$(NC)"
	@if [ -f /var/lib/dpkg/lock-frontend ] || [ -f /var/lib/dpkg/lock ]; then \
		echo "$(YELLOW)⚠ Found dpkg lock files:$(NC)"; \
		ls -la /var/lib/dpkg/lock* 2>/dev/null || true; \
		echo ""; \
		echo "$(CYAN)Checking if package manager is actually running...$(NC)"; \
		if pgrep -f "(apt|dpkg)" >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠ Package manager processes are running:$(NC)"; \
			ps aux | grep -E "(apt|dpkg)" | grep -v grep | head -5; \
			echo "$(YELLOW)⚠ Do NOT remove locks while processes are running!$(NC)"; \
			echo "$(YELLOW)⚠ Wait for them to finish or kill them first$(NC)"; \
		else \
			echo "$(GREEN)✓ No package manager processes running$(NC)"; \
			echo "$(GREEN)✓ Locks appear to be stale - safe to remove$(NC)"; \
			echo ""; \
			echo "$(YELLOW)To remove locks, run:$(NC)"; \
			echo "  sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock"; \
		fi; \
	elif [ -f /var/run/dnf.pid ]; then \
		echo "$(YELLOW)⚠ Found dnf lock:$(NC)"; \
		ls -la /var/run/dnf.pid 2>/dev/null || true; \
		if pgrep -f "(dnf|yum|rpm)" >/dev/null 2>&1; then \
			echo "$(YELLOW)⚠ Package manager is running - do not remove lock$(NC)"; \
		else \
			echo "$(GREEN)✓ Safe to remove: sudo rm -f /var/run/dnf.pid$(NC)"; \
		fi; \
	else \
		echo "$(GREEN)✓ No lock files found$(NC)"; \
	fi

# ============================================================================
# Source Build Commands
# ============================================================================

build: build-neurondb build-neuronagent build-neuronmcp build-neurondesktop ## Build all components from source
	@echo "$(CYAN)Copying binaries and configuration files to bin/ directory...$(NC)"
	@# Create directories for each component
	@mkdir -p bin/neurondb bin/neuronagent bin/neuronmcp bin/neurondesktop
	@# Copy NeuronDB extension files
	@echo "$(CYAN)[NeuronDB]$(NC)"
	@if [ -f NeuronDB/neurondb.control ]; then \
		cp NeuronDB/neurondb.control bin/neurondb/ && echo "  $(GREEN)✓$(NC) neurondb.control"; \
	fi
	@if [ -f NeuronDB/neurondb--1.0.sql ]; then \
		cp NeuronDB/neurondb--1.0.sql bin/neurondb/ && echo "  $(GREEN)✓$(NC) neurondb--1.0.sql"; \
	fi
	@if [ -f NeuronDB/neurondb--2.0.sql ]; then \
		cp NeuronDB/neurondb--2.0.sql bin/neurondb/ && echo "  $(GREEN)✓$(NC) neurondb--2.0.sql"; \
	fi
	@if [ -f NeuronDB/neurondb--1.0--2.0.sql ]; then \
		cp NeuronDB/neurondb--1.0--2.0.sql bin/neurondb/ && echo "  $(GREEN)✓$(NC) neurondb--1.0--2.0.sql"; \
	fi
	@if [ -f NeuronDB/neurondb.so ]; then \
		cp NeuronDB/neurondb.so bin/neurondb/ && echo "  $(GREEN)✓$(NC) neurondb.so"; \
	elif [ -f NeuronDB/neurondb.dylib ]; then \
		cp NeuronDB/neurondb.dylib bin/neurondb/ && echo "  $(GREEN)✓$(NC) neurondb.dylib"; \
	fi
	@# Copy NeuronAgent binary, configuration files, and setup SQL
	@echo "$(CYAN)[NeuronAgent]$(NC)"
	@if [ -f NeuronAgent/bin/neuronagent ]; then \
		cp NeuronAgent/bin/neuronagent bin/neuronagent/ && echo "  $(GREEN)✓$(NC) neuronagent binary"; \
	fi
	@if [ -f NeuronAgent/conf/neuronagent-config.yaml ]; then \
		cp NeuronAgent/conf/neuronagent-config.yaml bin/neuronagent/ && echo "  $(GREEN)✓$(NC) neuronagent-config.yaml"; \
	fi
	@if [ -f NeuronAgent/conf/agent-profiles.yaml ]; then \
		cp NeuronAgent/conf/agent-profiles.yaml bin/neuronagent/ && echo "  $(GREEN)✓$(NC) agent-profiles.yaml"; \
	fi
	@if [ -f NeuronAgent/bin/sql/neuron-agent.sql ]; then \
		cp NeuronAgent/bin/sql/neuron-agent.sql bin/neuronagent/ && echo "  $(GREEN)✓$(NC) neuron-agent.sql"; \
	elif [ -f NeuronAgent/neuron-agent.sql ]; then \
		cp NeuronAgent/neuron-agent.sql bin/neuronagent/ && echo "  $(GREEN)✓$(NC) neuron-agent.sql"; \
	fi
	@if [ -d NeuronAgent/bin/scripts ]; then \
		cp -r NeuronAgent/bin/scripts bin/neuronagent/ && echo "  $(GREEN)✓$(NC) scripts/"; \
	fi
	@if [ -d NeuronAgent/bin/conf ]; then \
		cp -r NeuronAgent/bin/conf bin/neuronagent/ && echo "  $(GREEN)✓$(NC) conf/"; \
	fi
	@# Copy NeuronMCP binary, configuration files, and setup SQL
	@echo "$(CYAN)[NeuronMCP]$(NC)"
	@if [ -f NeuronMCP/bin/neuronmcp ]; then \
		cp NeuronMCP/bin/neuronmcp bin/neuronmcp/ && echo "  $(GREEN)✓$(NC) neuronmcp binary"; \
	fi
	@if [ -f NeuronMCP/conf/mcp-config.json.example ]; then \
		cp NeuronMCP/conf/mcp-config.json.example bin/neuronmcp/ && echo "  $(GREEN)✓$(NC) mcp-config.json.example"; \
	fi
	@if [ -f NeuronMCP/bin/sql/neuron-mcp.sql ]; then \
		cp NeuronMCP/bin/sql/neuron-mcp.sql bin/neuronmcp/ && echo "  $(GREEN)✓$(NC) neuron-mcp.sql"; \
	elif [ -f NeuronMCP/sql/neuron-mcp.sql ]; then \
		cp NeuronMCP/sql/neuron-mcp.sql bin/neuronmcp/ && echo "  $(GREEN)✓$(NC) neuron-mcp.sql"; \
	elif [ -f NeuronMCP/neuron-mcp.sql ]; then \
		cp NeuronMCP/neuron-mcp.sql bin/neuronmcp/ && echo "  $(GREEN)✓$(NC) neuron-mcp.sql"; \
	fi
	@if [ -d NeuronMCP/bin/scripts ]; then \
		cp -r NeuronMCP/bin/scripts bin/neuronmcp/ && echo "  $(GREEN)✓$(NC) scripts/"; \
	fi
	@if [ -d NeuronMCP/bin/conf ]; then \
		cp -r NeuronMCP/bin/conf bin/neuronmcp/ && echo "  $(GREEN)✓$(NC) conf/"; \
	fi
	@# Copy NeuronDesktop binary and setup SQL
	@echo "$(CYAN)[NeuronDesktop]$(NC)"
	@if [ -f NeuronDesktop/bin/neurondesktop ]; then \
		cp NeuronDesktop/bin/neurondesktop bin/neurondesktop/ && echo "  $(GREEN)✓$(NC) neurondesktop binary"; \
	fi
	@if [ -f NeuronDesktop/bin/sql/neuron-desktop.sql ]; then \
		cp NeuronDesktop/bin/sql/neuron-desktop.sql bin/neurondesktop/ && echo "  $(GREEN)✓$(NC) neuron-desktop.sql"; \
	elif [ -f NeuronDesktop/bin/sql/neurondesktop.sql ]; then \
		cp NeuronDesktop/bin/sql/neurondesktop.sql bin/neurondesktop/ && echo "  $(GREEN)✓$(NC) neurondesktop.sql"; \
	elif [ -f NeuronDesktop/neuron-desktop.sql ]; then \
		cp NeuronDesktop/neuron-desktop.sql bin/neurondesktop/ && echo "  $(GREEN)✓$(NC) neuron-desktop.sql"; \
	fi
	@if [ -d NeuronDesktop/bin/scripts ]; then \
		cp -r NeuronDesktop/bin/scripts bin/neurondesktop/ && echo "  $(GREEN)✓$(NC) scripts/"; \
	elif [ -d NeuronDesktop/scripts ]; then \
		cp -r NeuronDesktop/scripts bin/neurondesktop/ && echo "  $(GREEN)✓$(NC) scripts/"; \
	fi
	@if [ -d NeuronDesktop/frontend/.next ]; then \
		cp -r NeuronDesktop/frontend/.next bin/neurondesktop/ && echo "  $(GREEN)✓$(NC) frontend/.next/"; \
	fi
	@if [ -f NeuronDesktop/frontend/package.json ]; then \
		cp NeuronDesktop/frontend/package.json bin/neurondesktop/frontend-package.json && echo "  $(GREEN)✓$(NC) frontend-package.json"; \
	fi
	@echo "$(GREEN)✓ All binaries and configuration files copied to bin/$(NC)"

build-neurondb: ## Build NeuronDB from source (uses build.sh)
	@echo "$(CYAN)Building NeuronDB from source...$(NC)"
	@if [ ! -f NeuronDB/build.sh ]; then \
		echo "$(YELLOW)Error: NeuronDB/build.sh not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(CYAN)Checking dependencies before build...$(NC)"
	@needs_install=0; \
	if ! command -v pg_config >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ pg_config not found - will install$(NC)"; \
		needs_install=1; \
	else \
		pg_ver=$$(pg_config --version 2>/dev/null | head -1 || echo ""); \
		echo "$(GREEN)✓ pg_config found$$([ -n "$$pg_ver" ] && echo " ($$pg_ver)" || echo "")$(NC)"; \
	fi; \
	if ! command -v make >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ make not found - will install$(NC)"; \
		needs_install=1; \
	else \
		echo "$(GREEN)✓ make found$(NC)"; \
	fi; \
	if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ C compiler not found - will install$(NC)"; \
		needs_install=1; \
	else \
		if command -v gcc >/dev/null 2>&1; then \
			echo "$(GREEN)✓ gcc found$(NC)"; \
		else \
			echo "$(GREEN)✓ clang found$(NC)"; \
		fi; \
	fi; \
	if [ $$needs_install -eq 0 ]; then \
		echo "$(GREEN)✓ All dependencies satisfied - skipping dependency installation$(NC)"; \
		echo "$(CYAN)Running build.sh with SKIP_DEP_INSTALL=1 (verbose mode)...$(NC)"; \
		echo "$(YELLOW)Note: You can see build progress below$(NC)"; \
		echo ""; \
		cd NeuronDB && chmod +x build.sh && export SKIP_DEP_INSTALL=1 && VERBOSE=1 ./build.sh; \
	else \
		echo "$(YELLOW)⚠ Some dependencies missing - build.sh will install them$(NC)"; \
		echo "$(CYAN)Running build.sh (will install missing dependencies, verbose mode)...$(NC)"; \
		echo "$(YELLOW)Note: You can see build progress below$(NC)"; \
		echo ""; \
		cd NeuronDB && chmod +x build.sh && VERBOSE=1 ./build.sh; \
	fi
	@echo "$(GREEN)✓ NeuronDB built$(NC)"

build-neurondb-now: ## Build NeuronDB immediately (skip deps check, force skip install, verbose)
	@echo "$(CYAN)Building NeuronDB immediately (skipping dependency installation)...$(NC)"
	@if [ ! -f NeuronDB/build.sh ]; then \
		echo "$(YELLOW)Error: NeuronDB/build.sh not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Using: SKIP_DEP_INSTALL=1 VERBOSE=1$(NC)"
	@echo "$(YELLOW)This will skip dependency installation and show verbose output$(NC)"
	@echo ""
	@cd NeuronDB && chmod +x build.sh && export SKIP_DEP_INSTALL=1 && VERBOSE=1 ./build.sh
	@echo "$(GREEN)✓ NeuronDB built$(NC)"

install-neurondb-now: ## Install NeuronDB immediately (skip deps check, force skip install, verbose)
	@echo "$(CYAN)Installing NeuronDB immediately (skipping dependency installation)...$(NC)"
	@if [ ! -f NeuronDB/build.sh ]; then \
		echo "$(YELLOW)Error: NeuronDB/build.sh not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Using: SKIP_DEP_INSTALL=1 VERBOSE=1 --skip-build$(NC)"
	@echo "$(YELLOW)This will skip dependency installation and show verbose output$(NC)"
	@echo ""
	@cd NeuronDB && chmod +x build.sh && export SKIP_DEP_INSTALL=1 && VERBOSE=1 ./build.sh --skip-build || $(MAKE) -C NeuronDB install
	@echo "$(GREEN)✓ NeuronDB installed$(NC)"

build-neuronagent: check-deps-neuronagent ## Build NeuronAgent from source
	@echo "$(CYAN)Building NeuronAgent from source...$(NC)"
	@cd NeuronAgent && $(MAKE) build
	@echo "$(GREEN)✓ NeuronAgent built$(NC)"

build-neuronmcp: check-deps-neuronmcp ## Build NeuronMCP from source
	@echo "$(CYAN)Building NeuronMCP from source...$(NC)"
	@cd NeuronMCP && $(MAKE) build
	@echo "$(GREEN)✓ NeuronMCP built$(NC)"

build-neurondesktop: check-deps-neurondesktop ## Build NeuronDesktop from source
	@echo "$(CYAN)Building NeuronDesktop from source...$(NC)"
	@cd NeuronDesktop && $(MAKE) build
	@echo "$(GREEN)✓ NeuronDesktop built$(NC)"

# ============================================================================
# Source Test Commands
# ============================================================================

test: test-neurondb test-neuronagent test-neuronmcp test-neurondesktop ## Run tests for all components

test-neurondb: ## Run NeuronDB tests
	@echo "$(CYAN)Running NeuronDB tests...$(NC)"
	@cd NeuronDB && $(MAKE) installcheck || echo "$(YELLOW)⚠ Some tests may have failed$(NC)"

test-neuronagent: ## Run NeuronAgent tests
	@echo "$(CYAN)Running NeuronAgent tests...$(NC)"
	@cd NeuronAgent && $(MAKE) test || echo "$(YELLOW)⚠ Some tests may have failed$(NC)"

test-neuronmcp: ## Run NeuronMCP tests
	@echo "$(CYAN)Running NeuronMCP tests...$(NC)"
	@cd NeuronMCP && $(MAKE) test || echo "$(YELLOW)⚠ Some tests may have failed$(NC)"

test-neurondesktop: ## Run NeuronDesktop tests
	@echo "$(CYAN)Running NeuronDesktop tests...$(NC)"
	@cd NeuronDesktop && $(MAKE) test || echo "$(YELLOW)⚠ Some tests may have failed$(NC)"

# ============================================================================
# Source Install Commands
# ============================================================================

install: install-neurondb install-neuronagent install-neuronmcp install-neurondesktop ## Install all components

install-neurondb: ## Install NeuronDB extension
	@echo "$(CYAN)Installing NeuronDB...$(NC)"
	@echo "$(CYAN)Checking dependencies before install...$(NC)"
	@needs_install=0; \
	if ! command -v pg_config >/dev/null 2>&1; then \
		echo "$(YELLOW)⚠ pg_config not found - will install dependencies$(NC)"; \
		needs_install=1; \
	else \
		echo "$(GREEN)✓ pg_config found$(NC)"; \
	fi; \
	if [ $$needs_install -eq 0 ]; then \
		echo "$(GREEN)✓ Dependencies satisfied - skipping dependency installation$(NC)"; \
		echo "$(CYAN)Running build.sh with --skip-build and SKIP_DEP_INSTALL=1...$(NC)"; \
		cd NeuronDB && SKIP_DEP_INSTALL=1 VERBOSE=1 ./build.sh --skip-build || $(MAKE) install; \
	else \
		echo "$(YELLOW)⚠ Dependencies missing - build.sh will install them$(NC)"; \
		cd NeuronDB && VERBOSE=1 ./build.sh --skip-build || $(MAKE) install; \
	fi
	@echo "$(GREEN)✓ NeuronDB installed$(NC)"
	@echo "$(YELLOW)Note: Add 'shared_preload_libraries = \"neurondb\"' to postgresql.conf$(NC)"

install-neuronagent: build-neuronagent ## Install NeuronAgent
	@echo "$(CYAN)Installing NeuronAgent...$(NC)"
	@echo "$(YELLOW)Note: NeuronAgent binary is in bin/neuronagent$(NC)"
	@echo "$(GREEN)✓ NeuronAgent ready$(NC)"

install-neuronmcp: build-neuronmcp ## Install NeuronMCP
	@echo "$(CYAN)Installing NeuronMCP...$(NC)"
	@echo "$(YELLOW)Note: NeuronMCP binary is in bin/neuronmcp$(NC)"
	@echo "$(GREEN)✓ NeuronMCP ready$(NC)"

install-neurondesktop: build-neurondesktop ## Install NeuronDesktop
	@echo "$(CYAN)Installing NeuronDesktop...$(NC)"
	@cd NeuronDesktop && $(MAKE) install
	@echo "$(YELLOW)Note: NeuronDesktop binary is in bin/neurondesktop$(NC)"
	@echo "$(GREEN)✓ NeuronDesktop installed$(NC)"

# ============================================================================
# Source Clean Commands
# ============================================================================

clean: clean-neurondb clean-neuronagent clean-neuronmcp clean-neurondesktop ## Clean all build artifacts
	@echo "$(CYAN)Cleaning root bin/ directory...$(NC)"
	@rm -rf bin/neuronagent bin/neuronmcp bin/neurondesktop bin/neurondb
	@echo "$(GREEN)✓ Root bin/ cleaned$(NC)"

clean-neurondb: ## Clean NeuronDB artifacts
	@echo "$(CYAN)Cleaning NeuronDB...$(NC)"
	@cd NeuronDB && $(MAKE) clean || true
	@echo "$(GREEN)✓ NeuronDB cleaned$(NC)"

clean-neuronagent: ## Clean NeuronAgent artifacts
	@echo "$(CYAN)Cleaning NeuronAgent...$(NC)"
	@cd NeuronAgent && $(MAKE) clean || true
	@echo "$(GREEN)✓ NeuronAgent cleaned$(NC)"

clean-neuronmcp: ## Clean NeuronMCP artifacts
	@echo "$(CYAN)Cleaning NeuronMCP...$(NC)"
	@cd NeuronMCP && $(MAKE) clean || true
	@echo "$(GREEN)✓ NeuronMCP cleaned$(NC)"

clean-neurondesktop: ## Clean NeuronDesktop artifacts
	@echo "$(CYAN)Cleaning NeuronDesktop...$(NC)"
	@cd NeuronDesktop && $(MAKE) clean || true
	@echo "$(GREEN)✓ NeuronDesktop cleaned$(NC)"
