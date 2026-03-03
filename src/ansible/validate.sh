#!/bin/bash
#
# Ansible Validation Script
# Validates Ansible playbooks and roles for common issues
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ERRORS=0
WARNINGS=0

echo "=== Ansible Validation ==="
echo ""

# Check if ansible-playbook is available
if ! command -v ansible-playbook &> /dev/null; then
    echo "⚠️  WARNING: ansible-playbook not found. Install Ansible to run full validation."
    echo "   Install with: pip install ansible"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✓ ansible-playbook found"
fi

# Check YAML syntax (if yaml module available)
echo ""
echo "=== Checking YAML Syntax ==="
if python3 -c "import yaml" 2>/dev/null; then
    for file in $(find . -name "*.yml" -o -name "*.yaml" | grep -v ".git"); do
        if python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null; then
            echo "✓ $file"
        else
            echo "✗ $file - YAML syntax error"
            ERRORS=$((ERRORS + 1))
        fi
    done
else
    echo "⚠️  Python yaml module not available, skipping YAML syntax check"
    echo "   Install with: pip install pyyaml"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for required files
echo ""
echo "=== Checking Required Files ==="
REQUIRED_FILES=(
    "ansible.cfg"
    "requirements.yml"
    "README.md"
    "inventory/hosts.yml"
    "playbooks/site.yml"
    "playbooks/infrastructure.yml"
    "playbooks/deploy-neurondb.yml"
    "group_vars/all.yml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file - Missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check for required roles
echo ""
echo "=== Checking Required Roles ==="
REQUIRED_ROLES=(
    "roles/common"
    "roles/postgresql"
    "roles/neurondb"
    "roles/neuronagent"
    "roles/neuronmcp"
    "roles/neurondesktop"
)

for role in "${REQUIRED_ROLES[@]}"; do
    if [ -d "$role" ] && [ -f "$role/tasks/main.yml" ]; then
        echo "✓ $role"
    else
        echo "✗ $role - Missing or incomplete"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check for common issues
echo ""
echo "=== Checking for Common Issues ==="

# Check for deprecated apt_key usage
if grep -r "apt_key:" roles/ playbooks/ 2>/dev/null | grep -v "keyring:" > /dev/null; then
    echo "⚠️  WARNING: Found apt_key without keyring (may be deprecated in newer Ansible)"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for undefined variables
if grep -r "{{.*}}" roles/ playbooks/ 2>/dev/null | grep -E "{{[^}]*\s+[^}]*}}" > /dev/null; then
    echo "⚠️  WARNING: Found potential undefined variables"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for missing no_log on sensitive tasks
if grep -r "password" roles/ playbooks/ 2>/dev/null | grep -v "no_log:" > /dev/null; then
    echo "⚠️  WARNING: Found password usage without no_log (security concern)"
    WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo ""
echo "=== Summary ==="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ All checks passed!"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo "✓ No errors found, but $WARNINGS warning(s)"
    exit 0
else
    echo "✗ Found $ERRORS error(s) and $WARNINGS warning(s)"
    exit 1
fi

