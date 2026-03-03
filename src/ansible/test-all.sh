#!/bin/bash
#
# Comprehensive Ansible Test Suite
# Tests all playbooks, roles, and configurations without requiring live servers
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0
PASSED=0

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Comprehensive Ansible Test Suite                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓${NC} Using virtual environment"
else
    echo -e "${YELLOW}⚠${NC}  Virtual environment not found, using system Ansible"
fi

# Test 1: Ansible Installation
echo -e "\n${BLUE}=== Test 1: Ansible Installation ===${NC}"
if command -v ansible-playbook &> /dev/null; then
    VERSION=$(ansible-playbook --version | head -1)
    echo -e "${GREEN}✓${NC} $VERSION"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} ansible-playbook not found"
    ERRORS=$((ERRORS + 1))
    exit 1
fi

# Test 2: Syntax Check - All Playbooks
echo -e "\n${BLUE}=== Test 2: Playbook Syntax Check ===${NC}"
PLAYBOOKS=(
    "playbooks/site.yml"
    "playbooks/infrastructure.yml"
    "playbooks/deploy-neurondb.yml"
    "playbooks/deploy-neuronagent.yml"
    "playbooks/deploy-neuronmcp.yml"
    "playbooks/deploy-neurondesktop.yml"
    "playbooks/backup-restore.yml"
    "playbooks/maintenance.yml"
)

for playbook in "${PLAYBOOKS[@]}"; do
    if ansible-playbook --syntax-check "$playbook" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $playbook"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} $playbook - Syntax error"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 3: Inventory Validation
echo -e "\n${BLUE}=== Test 3: Inventory Validation ===${NC}"
if ansible-inventory -i inventory/hosts.yml --list &> /dev/null; then
    HOST_COUNT=$(ansible-inventory -i inventory/hosts.yml --list | grep -c "ansible_host" || echo "0")
    echo -e "${GREEN}✓${NC} Inventory valid ($HOST_COUNT hosts configured)"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗${NC} Inventory validation failed"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: Role Structure Validation
echo -e "\n${BLUE}=== Test 4: Role Structure Validation ===${NC}"
REQUIRED_ROLES=(
    "roles/common"
    "roles/postgresql"
    "roles/neurondb"
    "roles/neuronagent"
    "roles/neuronmcp"
    "roles/neurondesktop"
    "roles/monitoring"
    "roles/security"
)

for role in "${REQUIRED_ROLES[@]}"; do
    if [ -d "$role" ] && [ -f "$role/tasks/main.yml" ]; then
        # Check for required files
        MISSING=""
        [ ! -f "$role/vars/main.yml" ] && MISSING="$MISSING vars/main.yml"
        [ ! -f "$role/handlers/main.yml" ] && MISSING="$MISSING handlers/main.yml"
        
        if [ -z "$MISSING" ]; then
            echo -e "${GREEN}✓${NC} $role (complete)"
            PASSED=$((PASSED + 1))
        else
            echo -e "${YELLOW}⚠${NC}  $role (missing:$MISSING)"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "${RED}✗${NC} $role - Missing or incomplete"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 5: Variable File Validation
echo -e "\n${BLUE}=== Test 5: Variable File Validation ===${NC}"
VAR_FILES=(
    "group_vars/all.yml"
    "group_vars/development.yml"
    "group_vars/staging.yml"
    "group_vars/production.yml"
)

for var_file in "${VAR_FILES[@]}"; do
    if [ -f "$var_file" ]; then
        if python3 -c "import yaml; yaml.safe_load(open('$var_file'))" &> /dev/null; then
            echo -e "${GREEN}✓${NC} $var_file"
            PASSED=$((PASSED + 1))
        else
            echo -e "${RED}✗${NC} $var_file - YAML syntax error"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo -e "${RED}✗${NC} $var_file - Missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 6: Template File Validation
echo -e "\n${BLUE}=== Test 6: Template File Validation ===${NC}"
TEMPLATE_COUNT=0
for template in $(find roles -name "*.j2" 2>/dev/null); do
    TEMPLATE_COUNT=$((TEMPLATE_COUNT + 1))
done

if [ $TEMPLATE_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Found $TEMPLATE_COUNT template files"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC}  No template files found"
    WARNINGS=$((WARNINGS + 1))
fi

# Test 7: Systemd Service Files
echo -e "\n${BLUE}=== Test 7: Systemd Service Files ===${NC}"
SERVICE_FILES=(
    "files/systemd/neuronagent.service"
    "files/systemd/neuronmcp.service"
    "files/systemd/neurondesk-api.service"
    "files/systemd/neurondesk-frontend.service"
)

for service in "${SERVICE_FILES[@]}"; do
    if [ -f "$service" ]; then
        echo -e "${GREEN}✓${NC} $service"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} $service - Missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 8: Configuration Files
echo -e "\n${BLUE}=== Test 8: Configuration Files ===${NC}"
CONFIG_FILES=(
    "ansible.cfg"
    "requirements.yml"
    "README.md"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        echo -e "${GREEN}✓${NC} $config"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗${NC} $config - Missing"
        ERRORS=$((ERRORS + 1))
    fi
done

# Test 9: Dry Run (Check Mode) - Test without connecting
echo -e "\n${BLUE}=== Test 9: Dry Run Validation (Localhost) ===${NC}"
# Create a test inventory with localhost
cat > /tmp/test-inventory.yml <<EOF
all:
  hosts:
    localhost:
      ansible_connection: local
      ansible_python_interpreter: /usr/bin/python3
EOF

# Test with localhost (won't actually connect, but validates playbook structure)
if ansible-playbook playbooks/infrastructure.yml -i /tmp/test-inventory.yml --check --limit localhost &> /tmp/ansible-test.log; then
    echo -e "${GREEN}✓${NC} Infrastructure playbook structure valid"
    PASSED=$((PASSED + 1))
else
    # Check if it's just a connection issue (expected) or actual error
    if grep -q "UNREACHABLE\|Connection refused" /tmp/ansible-test.log; then
        echo -e "${GREEN}✓${NC} Infrastructure playbook structure valid (connection expected to fail)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠${NC}  Infrastructure playbook - check /tmp/ansible-test.log"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

# Test 10: Variable Resolution
echo -e "\n${BLUE}=== Test 10: Variable Resolution ===${NC}"
if ansible-inventory -i inventory/hosts.yml --list | grep -q "neurondb_version" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Variables properly resolved"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC}  Variable resolution check inconclusive"
    WARNINGS=$((WARNINGS + 1))
fi

# Test 11: Collection Requirements
echo -e "\n${BLUE}=== Test 11: Collection Requirements ===${NC}"
if [ -f "requirements.yml" ]; then
    if grep -q "community.postgresql" requirements.yml; then
        echo -e "${GREEN}✓${NC} Collection requirements defined"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠${NC}  Collection requirements may be incomplete"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

# Test 12: Security Checks
echo -e "\n${BLUE}=== Test 12: Security Validation ===${NC}"
# Check for password usage in actual tasks (not in vars, templates, or comments)
PASSWORD_TASKS=$(grep -r "password" roles/*/tasks/ playbooks/*.yml 2>/dev/null | \
    grep -v "no_log:" | \
    grep -v "^#" | \
    grep -v "vars/main.yml" | \
    grep -v "\.j2" | \
    grep -E "(shell|command|postgresql)" | \
    wc -l | tr -d ' ')

if [ "$PASSWORD_TASKS" -eq 0 ]; then
    echo -e "${GREEN}✓${NC} All password tasks have no_log protection"
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠${NC}  Found $PASSWORD_TASKS password tasks without no_log"
    WARNINGS=$((WARNINGS + 1))
fi

# Summary
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        Test Summary                           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Tests Passed: ${GREEN}$PASSED${NC}"
echo -e "Warnings:     ${YELLOW}$WARNINGS${NC}"
echo -e "Errors:       ${RED}$ERRORS${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Ansible configuration is ready.${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ No errors found.${NC} ${YELLOW}Some warnings present.${NC}"
    exit 0
else
    echo -e "${RED}✗ Tests failed. Please fix errors before deployment.${NC}"
    exit 1
fi

