#!/bin/bash
#
# vagrant/provision-rocky.sh - Provision Rocky Linux VM for RPM package testing
#
# Installs PostgreSQL 18, tools, and configures the environment
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[PROVISION]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PROVISION]${NC} $*"
}

log_error() {
    echo -e "${RED}[PROVISION]${NC} $*" >&2
}

# Update system
log_info "Updating system packages..."
dnf update -y

# Install basic tools
log_info "Installing basic tools..."
dnf install -y \
    curl \
    wget \
    git \
    gcc \
    gcc-c++ \
    make \
    rpm-build \
    rpmdevtools \
    dnf-plugins-core

# Install PostgreSQL 18 from official repository
log_info "Installing PostgreSQL 18..."

# Add PostgreSQL repository
dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-9-x86_64/pgdg-redhat-repo-latest.noarch.rpm

# Install missing dependencies (try both names)
dnf install -y perl-IPC-Run3 || dnf install -y perl-IPC-Run || true

# Install PostgreSQL 18
dnf install -y --skip-broken postgresql18-server postgresql18 postgresql18-devel

# Initialize and start PostgreSQL
log_info "Initializing PostgreSQL 18..."
/usr/pgsql-18/bin/postgresql-18-setup initdb

# Configure PostgreSQL 18
log_info "Configuring PostgreSQL 18..."

PG_VERSION=18
PG_DATA="/var/lib/pgsql/${PG_VERSION}/data"
PG_CONF="${PG_DATA}/postgresql.conf"
PG_HBA="${PG_DATA}/pg_hba.conf"

# Allow connections from all IPs (for testing)
sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF" || \
sed -i "s/listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF" || true

# Add host-based authentication for testing
if ! grep -q "host all all 0.0.0.0/0 md5" "$PG_HBA"; then
    echo "host all all 0.0.0.0/0 md5" >> "$PG_HBA"
fi
if ! grep -q "host all all ::/0 md5" "$PG_HBA"; then
    echo "host all all ::/0 md5" >> "$PG_HBA"
fi

# Start and enable PostgreSQL
systemctl enable postgresql-18
systemctl start postgresql-18

# Wait for PostgreSQL to be ready
log_info "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if sudo -u postgres /usr/pgsql-18/bin/psql -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "PostgreSQL is ready"
        break
    fi
    sleep 1
done

# Create test database and user
log_info "Creating test database and user..."
sudo -u postgres /usr/pgsql-18/bin/psql <<EOF
-- Create test database
CREATE DATABASE neurondb_test;
CREATE DATABASE neurondb;

-- Create test user
CREATE USER neurondb_user WITH PASSWORD 'neurondb_test';
ALTER USER neurondb_user CREATEDB;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE neurondb_test TO neurondb_user;
GRANT ALL PRIVILEGES ON DATABASE neurondb TO neurondb_user;

-- Allow postgres user to connect
ALTER USER postgres WITH PASSWORD 'postgres';
EOF

log_success "Test database and user created"

# Install GitHub CLI (optional, for downloading packages)
log_info "Installing GitHub CLI..."
if ! command -v gh >/dev/null 2>&1; then
    dnf install -y 'dnf-command(config-manager)'
    dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
    dnf install -y gh
    log_success "GitHub CLI installed"
else
    log_success "GitHub CLI already installed"
fi

# Create directories for packages and results
log_info "Creating directories..."
mkdir -p /vagrant/packages
mkdir -p /vagrant/test-results
mkdir -p /home/vagrant/packages

# Set up environment for vagrant user
log_info "Setting up environment..."
cat >> /home/vagrant/.bashrc <<'EOF'
# NeuronDB testing environment
export PATH="/usr/local/bin:/usr/pgsql-18/bin:$PATH"
export PGHOST=localhost
export PGPORT=5432
export PGUSER=postgres
export PGPASSWORD=postgres
EOF

log_success "Provisioning completed successfully!"

