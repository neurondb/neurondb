# NeuronDB SDKs

Official client libraries for NeuronDB ecosystem components.

## Available SDKs

### Python
- **neuronagent** - NeuronAgent Python client
- Installation: `pip install neuronagent`
- Documentation: [Python SDK Docs](python/README.md)

### TypeScript/JavaScript
- **@neurondb/neuronagent** - NeuronAgent TypeScript client
- **@neurondb/neurondesktop** - NeuronDesktop TypeScript client
- Installation: `npm install @neurondb/neuronagent @neurondb/neurondesktop`
- Documentation: [TypeScript SDK Docs](typescript/README.md)

## Generating SDKs

SDKs are generated from OpenAPI specifications using OpenAPI Generator.

### Prerequisites

```bash
# Install OpenAPI Generator CLI
npm install -g @openapitools/openapi-generator-cli
```

### Generate All SDKs

```bash
./scripts/neurondb-pkgs.sh generate-sdk --language python
./scripts/neurondb-pkgs.sh generate-sdk --language javascript
```

### Generate Individual SDKs

```bash
# Python SDK
openapi-generator-cli generate \
  -i NeuronAgent/openapi/openapi.yaml \
  -g python \
  -o sdks/python/neuronagent

# TypeScript SDK
openapi-generator-cli generate \
  -i NeuronAgent/openapi/openapi.yaml \
  -g typescript-axios \
  -o sdks/typescript/neuronagent
```

## Examples

See the `examples/` directories in each SDK for usage examples:
- Basic agent usage
- RAG pipeline
- Vector search
- Multi-agent collaboration

## Contributing

When updating SDKs:
1. Update OpenAPI specifications
2. Regenerate SDKs using `neurondb-pkgs.sh generate-sdk`
3. Update examples if API changes
4. Run tests to verify compatibility

## Versioning

SDKs follow semantic versioning and are versioned independently:
- Major: Breaking API changes
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible








