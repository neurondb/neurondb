#!/bin/bash
# Setup crash capture for PostgreSQL debugging
# Run this before starting PostgreSQL

# Set ulimit for core dumps
ulimit -c unlimited

# Create core dump directory
sudo mkdir -p /tmp/core
sudo chmod 1777 /tmp/core

# Set core dump pattern (requires sudo)
sudo sysctl -w kernel.core_pattern="/tmp/core/core.%e.%p.%h.%t"

# Make it persistent
echo "kernel.core_pattern = /tmp/core/core.%e.%p.%h.%t" | sudo tee /etc/sysctl.d/99-core-dump.conf
sudo sysctl -p /etc/sysctl.d/99-core-dump.conf

# Verify
echo "=== Core dump settings ==="
ulimit -a | grep "core file size"
cat /proc/sys/kernel/core_pattern

echo ""
echo "=== PostgreSQL logging setup ==="
echo "Run these SQL commands:"
echo ""
cat << 'SQL'
ALTER SYSTEM SET log_min_messages = 'debug1';
ALTER SYSTEM SET log_error_verbosity = 'verbose';
ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';
ALTER SYSTEM SET logging_collector = on;
ALTER SYSTEM SET log_directory = 'log';
ALTER SYSTEM SET log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log';
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_duration = on;
SELECT pg_reload_conf();
SQL

