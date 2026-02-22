# Hub contributions

Copy the contents of `metrics/` into your neurondb-hub backend (or gateway) to expose Prometheus metrics at `/metrics`.

## metrics

- **Register()**: call once at startup (e.g. in main).
- **Middleware(next http.Handler)**: wrap your HTTP router to record request duration and count.
- **Handler()**: http.Handler for GET /metrics (Prometheus scrape).
- **CountAgentCreation()**, **CountKnowledgeIngest()**, **CountAuthFailure()**: increment business metrics.

Register and mount:

```go
import "your-hub-repo/internal/metrics"

func main() {
    metrics.Register()
    mux := http.NewServeMux()
    mux.Handle("/metrics", metrics.Handler())
    mux.Handle("/", metrics.Middleware(yourHandler))
    // ...
}
```
