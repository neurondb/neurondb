# High Availability Architecture

## Overview

This document describes the high availability (HA) architecture for NeuronDB ecosystem in production.

## Architecture Diagram

```
                    ┌─────────────┐
                    │   Load      │
                    │  Balancer   │
                    │  (Nginx)    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ Desktop │       │ Desktop │       │ Desktop │
   │  API 1  │       │  API 2  │       │  API 3  │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ Agent 1 │       │ Agent 2 │       │ Agent 3 │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
              ┌───────────▼───────────┐
              │  PostgreSQL Primary │
              │   (with Patroni)    │
              └───────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │Replica 1│       │Replica 2│       │Replica 3│
   └─────────┘       └─────────┘       └─────────┘
```

## Components

### 1. Load Balancer

**Nginx** or **HAProxy** for:
- Request distribution
- SSL termination
- Health checks
- Session affinity (if needed)

### 2. Application Layer

**Stateless Services** (can scale horizontally):
- NeuronDesktop API (2+ replicas)
- NeuronAgent (2+ replicas)
- NeuronDesktop Frontend (2+ replicas)

**Stateful Services**:
- NeuronMCP (1 replica, can be scaled if stateless)

### 3. Database Layer

**PostgreSQL HA** using **Patroni**:
- Primary node (read/write)
- 2+ replica nodes (read-only)
- Automatic failover
- Connection pooling (PgBouncer)

## Setup

### Step 1: PostgreSQL HA with Patroni

```yaml
# docker-compose.ha.yml
services:
  postgres-primary:
    image: postgres:17
    environment:
      PATRONI_SCOPE: neurondb
      PATRONI_NAME: postgres-primary
    volumes:
      - patroni-config:/etc/patroni
      - postgres-data:/var/lib/postgresql/data

  postgres-replica-1:
    image: postgres:17
    environment:
      PATRONI_SCOPE: neurondb
      PATRONI_NAME: postgres-replica-1
    depends_on:
      - postgres-primary

  patroni:
    image: patroni/patroni:latest
    environment:
      PATRONI_SCOPE: neurondb
      PATRONI_RESTAPI_LISTEN: 0.0.0.0:8008
```

### Step 2: Connection Pooling

```yaml
  pgbouncer:
    image: pgbouncer/pgbouncer:latest
    environment:
      DATABASES_HOST: postgres-primary
      DATABASES_PORT: 5432
      DATABASES_USER: neurondb
      DATABASES_PASSWORD: ${POSTGRES_PASSWORD}
      POOL_MODE: transaction
      MAX_CLIENT_CONN: 1000
      DEFAULT_POOL_SIZE: 25
```

### Step 3: Load Balancer

```nginx
# nginx.conf
upstream neurondesktop_api {
    least_conn;
    server neurondesk-api-1:8081;
    server neurondesk-api-2:8081;
    server neurondesk-api-3:8081;
}

upstream neuronagent {
    least_conn;
    server neuronagent-1:8080;
    server neuronagent-2:8080;
    server neuronagent-3:8080;
}

server {
    listen 80;
    server_name api.neurondb.example.com;

    location / {
        proxy_pass http://neurondesktop_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Failover Scenarios

### Database Primary Failure

1. Patroni detects primary failure
2. Elects new primary from replicas
3. Updates DNS/VIP to point to new primary
4. Applications reconnect automatically

### Application Node Failure

1. Load balancer detects health check failure
2. Removes node from pool
3. Traffic routed to healthy nodes
4. Auto-scaling can replace failed node

## Monitoring

### Health Checks

- Application: `/health` endpoint
- Database: `pg_isready`
- Load balancer: TCP/HTTP checks

### Metrics

- Request rate per node
- Error rate per node
- Database connection pool usage
- Replication lag

## Disaster Recovery

### Backup Strategy

- Daily full backups
- Continuous WAL archiving
- Off-site backup storage (S3)

### Recovery Time Objectives (RTO)

- Database failover: < 30 seconds
- Application recovery: < 5 minutes
- Full disaster recovery: < 1 hour

### Recovery Point Objectives (RPO)

- Database: < 5 minutes (WAL archiving)
- Application: Near-zero (stateless)

## Scaling

### Horizontal Scaling

- Add more application replicas
- Add more database replicas
- Use read replicas for queries

### Vertical Scaling

- Increase database resources
- Increase application resources
- Optimize queries and indexes

## Best Practices

1. **Use connection pooling**: PgBouncer for database connections
2. **Monitor replication lag**: Keep lag < 1 second
3. **Test failover regularly**: Monthly failover drills
4. **Use health checks**: All services should have health endpoints
5. **Implement circuit breakers**: Prevent cascade failures
6. **Use idempotent operations**: Handle retries gracefully








