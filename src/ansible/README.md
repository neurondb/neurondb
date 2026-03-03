# Ansible Infrastructure Provisioning for NeuronDB

This directory contains Ansible automation for infrastructure provisioning and deployment of the NeuronDB ecosystem across development, staging, and production environments.

## Overview

The Ansible automation complements existing Docker/Kubernetes scripts by handling:
- OS-level system configuration
- Infrastructure provisioning (firewall, security, system tuning)
- PostgreSQL installation and configuration
- NeuronDB extension build and installation
- Service deployment (NeuronAgent, NeuronMCP, NeuronDesktop)
- Backup and restore operations
- Maintenance tasks

## Quick Start

### Prerequisites

1. **Ansible Installation**
   ```bash
   # macOS
   brew install ansible

   # Ubuntu/Debian
   sudo apt-get install ansible

   # RHEL/CentOS/Rocky
   sudo yum install ansible
   ```

2. **SSH Access**
   - SSH key-based authentication to target hosts
   - Sudo/root access on target hosts

3. **Python 3**
   - Python 3.6+ required on target hosts

### Initial Setup

1. **Configure Inventory**
   ```bash
   # Edit inventory/hosts.yml with your server details
   vi inventory/hosts.yml
   ```

2. **Configure Variables**
   ```bash
   # Edit group_vars for your environment
   vi group_vars/production.yml
   ```

3. **Set Up Ansible Vault (for secrets)**
   ```bash
   # Create vault password file
   echo "your-vault-password" > .vault_pass
   chmod 600 .vault_pass

   # Encrypt sensitive variables
   ansible-vault encrypt group_vars/production.yml
   ```

### Basic Usage

#### Deploy Complete Ecosystem

```bash
# Deploy to all hosts in production
ansible-playbook playbooks/site.yml -i inventory/hosts.yml --limit production

# Deploy to specific host
ansible-playbook playbooks/site.yml -i inventory/hosts.yml --limit prod-neurondb-01

# Dry run (check mode)
ansible-playbook playbooks/site.yml -i inventory/hosts.yml --check
```

#### Deploy Individual Components

```bash
# Deploy only infrastructure
ansible-playbook playbooks/infrastructure.yml -i inventory/hosts.yml

# Deploy only NeuronDB
ansible-playbook playbooks/deploy-neurondb.yml -i inventory/hosts.yml

# Deploy only NeuronAgent
ansible-playbook playbooks/deploy-neuronagent.yml -i inventory/hosts.yml

# Deploy only NeuronMCP
ansible-playbook playbooks/deploy-neuronmcp.yml -i inventory/hosts.yml

# Deploy only NeuronDesktop
ansible-playbook playbooks/deploy-neurondesktop.yml -i inventory/hosts.yml
```

## Directory Structure

```
ansible/
├── ansible.cfg              # Ansible configuration
├── requirements.yml         # Ansible role dependencies
├── README.md               # This file
├── playbooks/              # Playbooks
│   ├── site.yml           # Main orchestration playbook
│   ├── infrastructure.yml  # Infrastructure provisioning
│   ├── deploy-*.yml        # Component deployment playbooks
│   ├── backup-restore.yml  # Backup and restore operations
│   └── maintenance.yml     # Maintenance tasks
├── roles/                  # Ansible roles
│   ├── common/            # Common system setup
│   ├── postgresql/        # PostgreSQL installation
│   ├── neurondb/          # NeuronDB extension
│   ├── neuronagent/       # NeuronAgent service
│   ├── neuronmcp/         # NeuronMCP service
│   ├── neurondesktop/     # NeuronDesktop service
│   ├── monitoring/        # Monitoring setup
│   └── security/          # Security hardening
├── inventory/              # Inventory files
│   ├── hosts.yml         # Host inventory
│   └── group_vars/        # Group-specific variables
├── group_vars/             # Environment variables
│   ├── all.yml           # Global variables
│   ├── development.yml    # Development environment
│   ├── staging.yml        # Staging environment
│   └── production.yml     # Production environment
└── files/                  # Static files
    ├── systemd/           # Systemd service files
    └── configs/          # Configuration templates
```

## Inventory Setup

### Basic Inventory

Edit `inventory/hosts.yml`:

```yaml
all:
  children:
    production:
      hosts:
        prod-neurondb-01:
          ansible_host: 10.0.1.10
          ansible_user: root
        prod-neurondb-02:
          ansible_host: 10.0.1.11
          ansible_user: root
```

### Host-Specific Variables

Create `inventory/host_vars/prod-neurondb-01.yml`:

```yaml
neurondb_maintenance_work_mem: "1GB"
postgresql_max_connections: 300
```

## Configuration

### Global Variables (`group_vars/all.yml`)

- Component versions
- Installation paths
- Service users
- Build options

### Environment Variables

- **Development** (`group_vars/development.yml`): Lower resources, debug enabled
- **Staging** (`group_vars/staging.yml`): Production-like configuration
- **Production** (`group_vars/production.yml`): Optimized for performance and security

### Secrets Management

Use Ansible Vault for sensitive data:

```bash
# Encrypt variables file
ansible-vault encrypt group_vars/production.yml

# Edit encrypted file
ansible-vault edit group_vars/production.yml

# View encrypted file
ansible-vault view group_vars/production.yml
```

## Playbooks

### site.yml

Main orchestration playbook that deploys the complete ecosystem:

```bash
ansible-playbook playbooks/site.yml -i inventory/hosts.yml
```

### infrastructure.yml

Provisions OS-level infrastructure:
- System packages
- Firewall configuration
- Security hardening
- System tuning
- SSL/TLS certificates

### deploy-neurondb.yml

Deploys PostgreSQL and NeuronDB extension:
- PostgreSQL installation
- Database configuration
- NeuronDB extension build and installation

### deploy-neuronagent.yml

Deploys NeuronAgent service:
- Binary build/installation
- Configuration
- Systemd service setup

### deploy-neuronmcp.yml

Deploys NeuronMCP service:
- Binary build/installation
- Configuration
- Systemd service setup

### deploy-neurondesktop.yml

Deploys NeuronDesktop (API and Frontend):
- API build/installation
- Frontend build/installation
- Systemd service setup

### backup-restore.yml

Database backup and restore operations:

```bash
# Backup
ansible-playbook playbooks/backup-restore.yml -i inventory/hosts.yml \
  --tags backup

# Restore
ansible-playbook playbooks/backup-restore.yml -i inventory/hosts.yml \
  --tags restore \
  -e backup_file_path=/backups/neurondb/backup.dump
```

### maintenance.yml

Maintenance operations:
- System updates
- Database VACUUM
- Log cleanup
- Backup cleanup

```bash
ansible-playbook playbooks/maintenance.yml -i inventory/hosts.yml
```

## Roles

### common

OS-level setup:
- Package installation
- User/group creation
- Directory structure
- System limits
- Log rotation

### postgresql

PostgreSQL installation and configuration:
- Repository setup
- PostgreSQL installation
- Configuration files
- Database and user creation

### neurondb

NeuronDB extension:
- Build dependencies
- Extension build
- Extension installation
- Verification

### neuronagent

NeuronAgent service:
- Go installation
- Binary build
- Configuration
- Systemd service

### neuronmcp

NeuronMCP service:
- Go installation
- Binary build
- Configuration
- Systemd service

### neurondesktop

NeuronDesktop service:
- Node.js installation
- Go installation
- API and frontend build
- Systemd services

## Integration with Existing Scripts

The Ansible playbooks integrate with existing shell scripts:

### neurondb-setup.sh

Used for package installation and setup verification.

### neurondb-database.sh

Used for backup and restore operations:
- `backup-restore.yml` calls `neurondb-database.sh backup`
- `backup-restore.yml` calls `neurondb-database.sh restore`

### neurondb-healthcheck.sh

Used for health verification:
- `deploy-neurondb.yml` calls `neurondb-healthcheck.sh quick`
- `deploy-neuronagent.yml` calls `neurondb-healthcheck.sh health`

## Common Tasks

### Deploy to New Server

```bash
# 1. Add server to inventory
vi inventory/hosts.yml

# 2. Test connectivity
ansible all -i inventory/hosts.yml -m ping

# 3. Deploy infrastructure
ansible-playbook playbooks/infrastructure.yml -i inventory/hosts.yml --limit new-server

# 4. Deploy NeuronDB
ansible-playbook playbooks/deploy-neurondb.yml -i inventory/hosts.yml --limit new-server

# 5. Deploy services
ansible-playbook playbooks/site.yml -i inventory/hosts.yml --limit new-server
```

### Update Configuration

```bash
# Update configuration and restart services
ansible-playbook playbooks/deploy-neuronagent.yml -i inventory/hosts.yml \
  -e neuronagent_log_level=debug
```

### Run Maintenance

```bash
# Run maintenance tasks
ansible-playbook playbooks/maintenance.yml -i inventory/hosts.yml
```

### Backup Database

```bash
# Create backup
ansible-playbook playbooks/backup-restore.yml -i inventory/hosts.yml \
  --tags backup
```

## Troubleshooting

### Connection Issues

```bash
# Test SSH connectivity
ansible all -i inventory/hosts.yml -m ping

# Test with verbose output
ansible all -i inventory/hosts.yml -m ping -vvv
```

### Permission Issues

```bash
# Use become with specific user
ansible-playbook playbooks/site.yml -i inventory/hosts.yml \
  --become --become-user=root
```

### Service Not Starting

```bash
# Check service status
ansible all -i inventory/hosts.yml -m systemd \
  -a "name=neuronagent state=started"

# View logs
ansible all -i inventory/hosts.yml -m shell \
  -a "journalctl -u neuronagent -n 50"
```

### Build Failures

```bash
# Check build dependencies
ansible all -i inventory/hosts.yml -m shell \
  -a "which gcc && which make && which pg_config"

# Check Go installation
ansible all -i inventory/hosts.yml -m shell \
  -a "go version"
```

## Best Practices

1. **Use Ansible Vault for Secrets**
   - Never commit unencrypted passwords
   - Use vault for database passwords, API keys

2. **Test in Development First**
   - Always test playbooks in development
   - Use `--check` mode for dry runs

3. **Version Control**
   - Commit inventory and playbooks
   - Use tags for versioning

4. **Idempotency**
   - Playbooks are idempotent (safe to re-run)
   - Use `--check` to preview changes

5. **Limit Scope**
   - Use `--limit` to target specific hosts
   - Use tags to run specific tasks

## Advanced Usage

### Using Tags

```bash
# Run only infrastructure tasks
ansible-playbook playbooks/infrastructure.yml --tags firewall

# Skip certain tasks
ansible-playbook playbooks/site.yml --skip-tags backup
```

### Parallel Execution

```bash
# Run on multiple hosts in parallel
ansible-playbook playbooks/site.yml -f 10
```

### Custom Variables

```bash
# Override variables at runtime
ansible-playbook playbooks/deploy-neurondb.yml \
  -e postgresql_version=18 \
  -e neurondb_version=2.1.0
```

## Security Considerations

1. **SSH Keys**: Use SSH key-based authentication
2. **Vault**: Encrypt all sensitive variables
3. **Firewall**: Enable firewall rules in production
4. **Users**: Use dedicated service users (not root)
5. **Permissions**: Restrict file permissions

## Support

For issues and questions:
- Check existing scripts in `scripts/` directory
- Review deployment documentation in `Docs/deployment/`
- See troubleshooting guide in `Docs/operations/troubleshooting.md`

## Related Documentation

- [Main README](../README.md)
- [Deployment Guide](../Docs/deployment/production-install.md)
- [Scripts Documentation](../scripts/README.md)
- [Contributing Guide](../CONTRIBUTING.md)

