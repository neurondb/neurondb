# NeuronAgent CLI Guide

Complete guide to using the NeuronAgent command-line interface for workflow management and agent operations.

## Installation

The CLI is included in the NeuronAgent repository. Build it with:

```bash
cd NeuronAgent/cli
go build -o neuronagent-cli .
```

Or install globally:

```bash
go install ./cli
```

## Configuration

Set your API URL and key:

```bash
export NEURONAGENT_URL=http://localhost:8080
export NEURONAGENT_API_KEY=your_api_key_here
```

Or use command-line flags:

```bash
neuronagent-cli --url http://localhost:8080 --key your_api_key workflow list
```

## Workflow Commands

### Create Workflow

Create a workflow from a YAML file:

```bash
neuronagent-cli workflow create workflow.yaml
```

**Example workflow.yaml:**

```yaml
name: data-pipeline
description: Process and analyze data
type: dag
steps:
  - id: fetch
    name: Fetch Data
    type: http
    config:
      url: https://api.example.com/data
  - id: process
    name: Process Data
    type: agent
    depends_on:
      - fetch
    config:
      agent_id: data-processor
  - id: store
    name: Store Results
    type: database
    depends_on:
      - process
    config:
      table: results
```

### List Workflows

List all workflows:

```bash
neuronagent-cli workflow list
```

With JSON output:

```bash
neuronagent-cli workflow list --format json
```

### Show Workflow Details

Display detailed information about a workflow:

```bash
neuronagent-cli workflow show <workflow-id>
```

### Export Workflow

Export a workflow to YAML format:

```bash
neuronagent-cli workflow export <workflow-id> > exported_workflow.yaml
```

### Validate Workflow

Validate a workflow definition without creating it:

```bash
neuronagent-cli workflow validate workflow.yaml
```

### List Workflow Templates

View available workflow templates:

```bash
neuronagent-cli workflow templates
```

Available templates:
- `data-pipeline`: Data processing pipeline
- `customer-support`: Customer support automation
- `document-qa`: Document Q&A workflow
- `code-reviewer`: Code review automation
- `research-assistant`: Research workflow
- `report-generator`: Report generation workflow

## Agent Commands

### Create Agent

```bash
neuronagent-cli create --config agent.yaml
```

### List Agents

```bash
neuronagent-cli list
```

### Show Agent

```bash
neuronagent-cli show <agent-id>
```

### Update Agent

```bash
neuronagent-cli update <agent-id> --config updated_agent.yaml
```

### Delete Agent

```bash
neuronagent-cli delete <agent-id>
```

### Clone Agent

```bash
neuronagent-cli clone <agent-id> --name new-agent-name
```

## Testing Commands

### Test Agent

Test an agent with a message:

```bash
neuronagent-cli test <agent-id> --message "Hello, how are you?"
```

### Test with Workflow

Test an agent with workflow execution:

```bash
neuronagent-cli test <agent-id> --workflow --debug
```

## Output Formats

All commands support multiple output formats:

- **Text** (default): Human-readable format
- **JSON**: Machine-readable JSON format

```bash
neuronagent-cli workflow list --format json
```

## Error Handling

The CLI provides clear error messages:

```bash
Error: API key is required. Set NEURONAGENT_API_KEY environment variable or use --key flag
```

Common errors:
- **401 Unauthorized**: Invalid or missing API key
- **404 Not Found**: Resource doesn't exist
- **400 Bad Request**: Invalid input or configuration
- **500 Internal Server Error**: Server error (check server logs)

## Examples

### Complete Workflow Example

```bash
# 1. Create workflow
neuronagent-cli workflow create my_workflow.yaml

# 2. List workflows to get ID
neuronagent-cli workflow list

# 3. Show workflow details
neuronagent-cli workflow show <workflow-id>

# 4. Export workflow
neuronagent-cli workflow export <workflow-id> > backup.yaml

# 5. Validate before updating
neuronagent-cli workflow validate updated_workflow.yaml
```

### Agent Management Example

```bash
# Create agent from template
neuronagent-cli create --template customer-support --name support-bot

# List all agents
neuronagent-cli list

# Test the agent
neuronagent-cli test <agent-id> --message "Help me with my order"

# Update agent configuration
neuronagent-cli update <agent-id> --config new_config.yaml
```

## Troubleshooting

### Connection Issues

**Error: "Failed to connect to API"**
- Verify server is running: `curl http://localhost:8080/health`
- Check API URL is correct
- Verify network connectivity

### Authentication Issues

**Error: "Invalid API key"**
- Generate new API key: `go run cmd/generate-key/main.go`
- Verify key is set in environment or flag
- Check key has necessary permissions

### Workflow Issues

**Error: "Workflow validation failed"**
- Check YAML syntax
- Verify all required fields are present
- Check for circular dependencies in steps
- Ensure step IDs are unique

### CLI Issues

**Error: "Command not found"**
- Ensure CLI is in PATH
- Or use full path: `./neuronagent-cli`
- Rebuild if needed: `go build -o neuronagent-cli .`

## Best Practices

1. **Use Configuration Files**: Store workflow and agent configs in version control
2. **Validate Before Creating**: Always validate workflows before creating
3. **Use Templates**: Start with templates for common use cases
4. **Export for Backup**: Regularly export workflows for backup
5. **JSON for Automation**: Use JSON format for scripting and automation
6. **Error Handling**: Always check command exit codes in scripts

