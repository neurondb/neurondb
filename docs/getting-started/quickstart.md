# NeuronDB Quickstart (5 minutes)

1. **Prerequisites**: Docker and Docker Compose, or a PostgreSQL 16+ server.

2. **Start with Docker Compose** (from repo root):
   ```bash
   docker compose up -d neurondb neuronagent neurondesktop-api
   ```
   Wait for health checks (e.g. 30s). NeuronDB on port 5433, NeuronAgent on 8080, NeuronDesktop API on 8081.

3. **Create extension and run a query**:
   ```bash
   psql -h localhost -p 5433 -U neurondb -d neurondb -c "CREATE EXTENSION IF NOT EXISTS neurondb; SELECT neurondb.version();"
   ```

4. **Train a simple model** (optional):
   ```sql
   CREATE TABLE train_data (x float, y float);
   INSERT INTO train_data SELECT i::float, 2*i::float + 1 FROM generate_series(1,100) i;
   SELECT neurondb.train('default', 'linear_regression', 'train_data', 'y', ARRAY['x'], '{}'::jsonb);
   ```

5. **Open NeuronDesktop** (if frontend is running): http://localhost:3000 and connect to the API at http://localhost:8081.

For production, set strong passwords via environment variables, enable TLS, and follow [production-install](../deployment/production-install.md).
