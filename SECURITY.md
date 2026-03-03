# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in NeuronDB, please report it by emailing:

**support@neurondb.ai**

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Time

- We aim to acknowledge receipt within 48 hours
- We will investigate and provide an initial assessment within 7 days
- We will work with you to understand and resolve the issue

### Disclosure Policy

- Please do not publicly disclose the vulnerability until we have released a fix
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We follow responsible disclosure practices

## Security Best Practices

When using NeuronDB:

1. **Access Control**: Use PostgreSQL role-based access control
2. **Encryption**: Enable SSL/TLS for all connections
3. **Updates**: Keep PostgreSQL and NeuronDB up to date
4. **Auditing**: Enable query logging for sensitive operations
5. **Validation**: Validate all user inputs before vector operations

## Known Security Considerations

- Vector encryption features are for demonstration; use PostgreSQL's native encryption for production
- HTTP/LLM integration requires secure credential management
- Shared memory buffers should be sized appropriately to prevent DoS

Thank you for helping keep NeuronDB secure!

