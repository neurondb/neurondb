#!/bin/bash
# Setup script for NeuronAgent examples
# This script helps set up the environment and run examples

set -e

echo "=========================================="
echo "NeuronAgent Examples Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found${NC}"

# Check if Go is installed (for Go example)
if command -v go &> /dev/null; then
    echo -e "${GREEN}✅ Go found${NC}"
    GO_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Go not found (Go example will be skipped)${NC}"
    GO_AVAILABLE=false
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
fi

# Check if API key is set
if [ -z "$NEURONAGENT_API_KEY" ]; then
    echo ""
    echo -e "${YELLOW}⚠️  NEURONAGENT_API_KEY not set${NC}"
    echo "You can:"
    echo "  1. Set it manually: export NEURONAGENT_API_KEY=your_key"
    echo "  2. Generate one using: ../scripts/generate_api_keys.sh"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✅ API key is set${NC}"
fi

# Check if server is running
echo ""
echo "Checking if NeuronAgent server is running..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server is running${NC}"
else
    echo -e "${RED}❌ Server is not responding${NC}"
    echo "Make sure NeuronAgent server is running:"
    echo "  cd .. && go run cmd/agent-server/main.go"
    echo "  or"
    echo "  cd ../docker && docker compose up -d"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run examples:"
echo "  Python: python3 python_client.py"
if [ "$GO_AVAILABLE" = true ]; then
    echo "  Go:     go run go_client.go"
fi
echo ""
echo "Make sure NEURONAGENT_API_KEY is set:"
echo "  export NEURONAGENT_API_KEY=your_api_key"
echo ""

