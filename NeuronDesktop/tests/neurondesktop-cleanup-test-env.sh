#!/bin/bash
# Cleanup test environment for NeuronDesktop tests

set -e

echo "Cleaning up test environment..."

# Set default test database settings
export TEST_DB_HOST=${TEST_DB_HOST:-localhost}
export TEST_DB_PORT=${TEST_DB_PORT:-5432}
export TEST_DB_USER=${TEST_DB_USER:-neurondesk}
export TEST_DB_PASSWORD=${TEST_DB_PASSWORD:-neurondesk}
export TEST_DB_NAME=${TEST_DB_NAME:-neurondesk_test}

# Option to drop test database
if [ "${DROP_TEST_DB:-false}" = "true" ]; then
    echo "Dropping test database: $TEST_DB_NAME"
    PGPASSWORD=$TEST_DB_PASSWORD psql -h "$TEST_DB_HOST" -p "$TEST_DB_PORT" -U "$TEST_DB_USER" -d postgres -c "DROP DATABASE IF EXISTS $TEST_DB_NAME" || {
        echo "Warning: Failed to drop test database (may not exist)"
    }
    echo "Test database dropped."
else
    echo "Test database preserved. Set DROP_TEST_DB=true to drop it."
fi

echo "Cleanup complete."








