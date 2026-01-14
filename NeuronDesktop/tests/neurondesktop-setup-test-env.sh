#!/bin/bash
# Setup test environment for NeuronDesktop tests

set -e

echo "Setting up test environment..."

# Check if PostgreSQL is available
if ! command -v psql &> /dev/null; then
    echo "Error: psql not found. Please install PostgreSQL client tools."
    exit 1
fi

# Set default test database settings
export TEST_DB_HOST=${TEST_DB_HOST:-localhost}
export TEST_DB_PORT=${TEST_DB_PORT:-5432}
export TEST_DB_USER=${TEST_DB_USER:-neurondesk}
export TEST_DB_PASSWORD=${TEST_DB_PASSWORD:-neurondesk}
export TEST_DB_NAME=${TEST_DB_NAME:-neurondesk_test}

echo "Test database configuration:"
echo "  Host: $TEST_DB_HOST"
echo "  Port: $TEST_DB_PORT"
echo "  User: $TEST_DB_USER"
echo "  Database: $TEST_DB_NAME"

# Test connection to postgres database
PGPASSWORD=$TEST_DB_PASSWORD psql -h "$TEST_DB_HOST" -p "$TEST_DB_PORT" -U "$TEST_DB_USER" -d postgres -c "SELECT 1" > /dev/null 2>&1 || {
    echo "Error: Cannot connect to PostgreSQL. Please check your connection settings."
    exit 1
}

echo "PostgreSQL connection successful."

# Create test database if it doesn't exist
PGPASSWORD=$TEST_DB_PASSWORD psql -h "$TEST_DB_HOST" -p "$TEST_DB_PORT" -U "$TEST_DB_USER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$TEST_DB_NAME'" | grep -q 1 || {
    echo "Creating test database: $TEST_DB_NAME"
    PGPASSWORD=$TEST_DB_PASSWORD psql -h "$TEST_DB_HOST" -p "$TEST_DB_PORT" -U "$TEST_DB_USER" -d postgres -c "CREATE DATABASE $TEST_DB_NAME"
    echo "Test database created."
}

echo "Test environment setup complete."








