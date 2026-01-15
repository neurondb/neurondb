# Chaos Engineering Framework

This directory contains chaos engineering tests for NeuronAgent to ensure resilience and reliability.

## Test Scenarios

### Network Partition
- Simulates network partitions
- Tests circuit breaker behavior
- Verifies failover mechanisms

### Database Failure
- Tests database connection failures
- Verifies retry logic
- Tests error recovery

### LLM API Failure
- Tests LLM API rate limiting
- Tests LLM API timeouts
- Verifies circuit breaker for LLM calls

### Resource Exhaustion
- Tests memory exhaustion
- Tests CPU exhaustion
- Tests connection pool exhaustion

### Failover
- Tests primary node failure
- Tests replica promotion
- Tests health check recovery

## Running Tests

```bash
go test ./tests/chaos/... -v
```

## Adding New Tests

1. Create test function following naming convention `Test<Scenario>`
2. Use circuit breakers and error handlers from reliability package
3. Verify expected behavior after failures
4. Document test scenario in this README


