# Rate Limiting and Resilience

## Distributed rate limiting (Redis)

- NeuronAgent currently uses in-memory rate limiting per API key. For multi-replica deployments, use Redis-backed rate limiting so limits are shared across instances.
- Implementation: use a Redis key per API key (e.g. `ratelimit:{api_key_id}`) with INCR and EXPIRE, or use a library (e.g. go-redis with a sliding window). Check the limit before processing the request and return 429 when exceeded.
- Configure Redis via `REDIS_ADDR` (or similar); when set, enable Redis-backed rate limiter in AuthMiddleware.

## Retry logic for LLM API calls

- NeuronAgent LLM client retries on retryable errors (timeout, rate limit, 502/503/429, connection refused) with exponential backoff (500ms, 1s, 2s, … cap 30s), up to 3 attempts.
- For direct HTTP LLM providers (if added), use the same pattern: retry with backoff and respect Retry-After when present.

## Circuit breaker for external services

- For outbound HTTP calls (LLM, embeddings, web search), wrap the client with a circuit breaker: after N failures in a window, open the circuit and fail fast; after a cooldown, half-open and probe once.
- Use a library (e.g. sony/gobreaker) or implement a simple state machine (closed → open → half-open). Apply per destination (e.g. per LLM endpoint).

## Graceful shutdown

- See [graceful-shutdown.md](graceful-shutdown.md). All services support SIGTERM and drain in-flight requests before exit.
