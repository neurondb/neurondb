.PHONY: build clean test run build-client

build:
	go build -o bin/neurondb-mcp ./cmd/neurondb-mcp

build-client:
	go build -o bin/neurondb-mcp-client ./cmd/neurondb-mcp-client

clean:
	rm -rf bin/

test:
	go test ./...

run: build
	./bin/neurondb-mcp

