# Graceful Shutdown and Connection Draining

## NeuronAgent

- The server listens for `SIGINT`/`SIGTERM` and calls `srv.Shutdown(ctx)` with a timeout (e.g. 30s). In-flight requests complete; new requests receive 503 or connection close.
- Ensure DB connection pool is closed after HTTP server shutdown so no new queries start during drain.

## NeuronDesktop API

- Use `http.Server.Shutdown(context.WithTimeout(...))` in the main process. Stop accepting new connections and wait for active handlers to finish.
- Close database and Redis connections after the HTTP server has shut down.

## NeuronMCP

- If running as an HTTP server, implement the same pattern: catch SIGTERM, call Shutdown, then exit. For stdio mode, close stdin and wait for in-flight work.

## NeuronDB (PostgreSQL)

- PostgreSQL handles SIGTERM with a graceful shutdown: it stops accepting new connections and waits for existing sessions to finish (up to `terminate_after` or default). No application change required; ensure orchestrator gives a sufficient termination grace period (e.g. 60s).

## Kubernetes

- Set `terminationGracePeriodSeconds` (e.g. 30–60) on deployments so pods have time to drain. Use a preStop hook only if you need to deregister from a load balancer; otherwise, readiness probe will stop traffic when the pod is terminating.
