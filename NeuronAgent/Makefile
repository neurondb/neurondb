.PHONY: build test clean run migrate docker-build docker-up docker-down

# Build the application
build:
	@echo "Building NeuronAgent..."
	@mkdir -p bin
	@go build -o bin/neuronagent cmd/agent-server/main.go
	@echo "Copying all runtime files to bin/..."
	@mkdir -p bin/scripts bin/conf bin/scripts/lib bin/sql
	@cp scripts/neuronagent-*.sh bin/scripts/ 2>/dev/null || true
	@cp -r scripts/lib/*.sh bin/scripts/lib/ 2>/dev/null || true
	@cp -r conf/*.yaml bin/conf/ 2>/dev/null || true
	@cp neuron-agent.sql bin/sql/ 2>/dev/null || true
	@chmod +x bin/scripts/*.sh 2>/dev/null || true
	@echo "# NeuronAgent Runtime Package" > bin/README.md
	@echo "" >> bin/README.md
	@echo "This directory contains everything needed to run NeuronAgent." >> bin/README.md
	@echo "" >> bin/README.md
	@echo "## Structure" >> bin/README.md
	@echo "- neuronagent - Main binary executable" >> bin/README.md
	@echo "- scripts/ - Runtime utility scripts" >> bin/README.md
	@echo "- conf/ - Configuration files" >> bin/README.md
	@echo "- sql/ - Database schema files" >> bin/README.md
	@echo "" >> bin/README.md
	@echo "## Usage" >> bin/README.md
	@echo "Run: ./neuronagent" >> bin/README.md
	@echo "Setup: ./scripts/neuronagent-setup.sh" >> bin/README.md
	@echo "✓ Binary, scripts, configs, and SQL files copied to bin/ (complete runtime package)"

# Run tests with race detector (default - recommended for all testing)
test:
	@echo "Running tests with race detector..."
	@go test -v -race ./...

# Run tests without race detector (faster, less thorough)
test-fast:
	@echo "Running tests (fast mode, no race detector)..."
	@go test -v ./...

# Run tests with coverage (includes race detector)
test-coverage:
	@echo "Running tests with coverage and race detector..."
	@go test -v -race -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@rm -rf bin/ dist/ coverage.out coverage.html
	@echo "Note: Scripts in scripts/ and configs in conf/ are preserved"

# Run the application
run: build
	@echo "Running NeuronAgent..."
	@./bin/neuronagent

# Run database migrations
migrate:
	@echo "Running migrations..."
	@./scripts/neuronagent-migrate.sh

# Format code
fmt:
	@echo "Formatting code..."
	@go fmt ./...

# Lint code
lint:
	@echo "Linting code..."
	@golangci-lint run

# Docker build
docker-build:
	@echo "Building Docker image..."
	@docker build -t neuronagent:latest -f docker/Dockerfile .

# Docker compose up
docker-up:
	@echo "Starting Docker Compose..."
	@docker-compose -f docker/docker-compose.yml up -d

# Docker compose down
docker-down:
	@echo "Stopping Docker Compose..."
	@docker-compose -f docker/docker-compose.yml down

# Install dependencies
deps:
	@echo "Downloading dependencies..."
	@go mod download
	@go mod tidy

# Generate API keys
generate-key:
	@echo "Generating API key..."
	@./scripts/neuronagent-generate-keys.sh

