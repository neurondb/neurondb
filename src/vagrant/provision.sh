#!/bin/bash
#
# vagrant/provision.sh - Provision Vagrant VM for package testing
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
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get upgrade -y

# Install basic tools
log_info "Installing basic tools..."
apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common

# Install package inspection tools
log_info "Installing package inspection tools..."
apt-get install -y \
    dpkg-dev \
    fakeroot \
    debhelper

# Install PostgreSQL 18 from official repository
log_info "Installing PostgreSQL 18..."
apt-get install -y curl ca-certificates
install -d /usr/share/postgresql-common/pgdg
curl -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc --fail https://www.postgresql.org/media/keys/ACCC4CF8.asc
sh -c 'echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
apt-get update
apt-get install -y postgresql-18 postgresql-server-dev-18 postgresql-client-18

# Configure PostgreSQL 18
log_info "Configuring PostgreSQL 18..."

# Start PostgreSQL
systemctl start postgresql@18-main || systemctl start postgresql
systemctl enable postgresql@18-main || systemctl enable postgresql

# Configure PostgreSQL to listen on all interfaces (for testing)
PG_VERSION=18
if [ -d "/etc/postgresql/${PG_VERSION}/main" ]; then
    PG_CONF="/etc/postgresql/${PG_VERSION}/main/postgresql.conf"
    PG_HBA="/etc/postgresql/${PG_VERSION}/main/pg_hba.conf"
elif [ -f "/etc/postgresql/postgresql.conf" ]; then
    PG_CONF="/etc/postgresql/postgresql.conf"
    PG_HBA="/etc/postgresql/pg_hba.conf"
else
    # Try to find the config
    PG_CONF=$(sudo -u postgres psql -t -c "SHOW config_file;" 2>/dev/null | xargs || echo "")
    PG_HBA=$(sudo -u postgres psql -t -c "SHOW hba_file;" 2>/dev/null | xargs || echo "")
fi

if [ -n "$PG_CONF" ] && [ -f "$PG_CONF" ]; then
    # Allow connections from all IPs (for testing)
    sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF" || true
    sed -i "s/listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF" || true
fi

if [ -n "$PG_HBA" ] && [ -f "$PG_HBA" ]; then
    # Add host-based authentication for testing
    if ! grep -q "host all all 0.0.0.0/0 md5" "$PG_HBA"; then
        echo "host all all 0.0.0.0/0 md5" >> "$PG_HBA"
    fi
    if ! grep -q "host all all ::/0 md5" "$PG_HBA"; then
        echo "host all all ::/0 md5" >> "$PG_HBA"
    fi
fi

# Restart PostgreSQL to apply changes
systemctl restart postgresql@18-main || systemctl restart postgresql

# Wait for PostgreSQL to be ready
log_info "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if sudo -u postgres psql -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "PostgreSQL is ready"
        break
    fi
    sleep 1
done

# Create test database and user
log_info "Creating test database and user..."
sudo -u postgres psql <<EOF
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
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | \
        dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | \
        tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update
    apt-get install -y gh
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
export PATH="/usr/local/bin:$PATH"
export PGHOST=localhost
export PGPORT=5432
export PGUSER=postgres
export PGPASSWORD=postgres
EOF

log_success "Provisioning completed successfully!"

