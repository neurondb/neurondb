# NeuronDesktop Deployment Guide

## Prerequisites

- Docker and Docker Compose
- PostgreSQL 16+ (if not using Docker)
- Go 1.23+ (for building from source)
- Node.js 20+ (for building frontend)

## Quick Start with Docker

1. Clone the repository:
```bash
git clone <repository-url>
cd NeuronDesktop
```

2. Start all services:
```bash
docker-compose up -d
```

3. Initialize database:
```bash
docker-compose exec postgres psql -U neurondesk -d neurondesk -f /docker-entrypoint-initdb.d/001_initial_schema.sql
```

4. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8081
- Health Check: http://localhost:8081/health

## Manual Deployment

### Backend

1. Set environment variables:
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=neurondesk
export DB_PASSWORD=neurondesk
export DB_NAME=neurondesk
export SERVER_PORT=8081
export LOG_LEVEL=info
```

2. Initialize database:
```bash
createdb neurondesk
psql -d neurondesk -f api/migrations/001_initial_schema.sql
```

3. Build and run:
```bash
cd api
go mod download
go build -o neurondesk-api ./cmd/server
./neurondesk-api
```

### Frontend

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Set environment variables:
```bash
export NEXT_PUBLIC_API_URL=http://localhost:8081/api/v1
```

3. Build and run:
```bash
npm run build
npm start
```

## Production Deployment

### Backend

1. Build Docker image:
```bash
cd api
docker build -t neurondesk-api:latest .
```

2. Run with production settings:
```bash
docker run -d \
  -p 8081:8081 \
  -e DB_HOST=postgres.example.com \
  -e DB_PASSWORD=secure_password \
  -e LOG_LEVEL=warn \
  neurondesk-api:latest
```

### Frontend

1. Build Docker image:
```bash
cd frontend
docker build -t neurondesk-frontend:latest .
```

2. Run with production settings:
```bash
docker run -d \
  -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=https://api.example.com/api/v1 \
  neurondesk-frontend:latest
```

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `localhost` | Database hostname |
| `DB_PORT` | `5432` | Database port |
| `DB_USER` | `neurondesk` | Database username |
| `DB_PASSWORD` | `neurondesk` | Database password |
| `DB_NAME` | `neurondesk` | Database name |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8081` | Server port |
| `LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `LOG_FORMAT` | `json` | Log format (json, text) |
| `CORS_ALLOWED_ORIGINS` | `*` | Comma-separated list of allowed origins |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8081/api/v1` | Backend API URL |

## Security Considerations

1. **API Keys**: Store API keys securely. Never commit them to version control.

2. **Database**: Use strong passwords and enable SSL for production.

3. **CORS**: Configure `CORS_ALLOWED_ORIGINS` to restrict access.

4. **Rate Limiting**: Adjust rate limits based on your needs.

5. **HTTPS**: Use HTTPS in production with proper SSL certificates.

## Monitoring

### Health Checks

The backend provides a health check endpoint:
```bash
curl http://localhost:8081/health
```

### Metrics

View application metrics:
```bash
curl http://localhost:8081/api/v1/metrics
```

### Logs

View Docker logs:
```bash
docker-compose logs -f neurondesk-api
docker-compose logs -f neurondesk-frontend
```

## Scaling

### Horizontal Scaling

1. **Backend**: Run multiple instances behind a load balancer
2. **Frontend**: Use a CDN or static hosting
3. **Database**: Use connection pooling and read replicas

### Vertical Scaling

1. Increase database connection pool size
2. Adjust rate limits
3. Increase server resources

## Troubleshooting

### Database Connection Issues

```bash
# Test database connection
psql -h localhost -p 5432 -U neurondesk -d neurondesk

# Check database logs
docker-compose logs postgres
```

### API Not Responding

```bash
# Check API health
curl http://localhost:8081/health

# Check API logs
docker-compose logs neurondesk-api
```

### Frontend Not Loading

```bash
# Check frontend logs
docker-compose logs neurondesk-frontend

# Verify API URL
echo $NEXT_PUBLIC_API_URL
```

## Backup and Recovery

### Database Backup

```bash
# Backup
pg_dump -U neurondesk neurondesk > backup.sql

# Restore
psql -U neurondesk neurondesk < backup.sql
```

### Docker Volume Backup

```bash
# Backup
docker run --rm -v neurondesk_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data

# Restore
docker run --rm -v neurondesk_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /
```

