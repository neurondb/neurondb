# TLS for Inter-Service and Client Communication

## Overview

- **Client → services**: Use a reverse proxy (e.g. nginx, Traefik) or load balancer that terminates TLS and forwards to services on HTTP (or use TLS to the service).
- **Inter-service**: In Kubernetes, use in-cluster TLS (e.g. cert-manager with internal Issuer) or mTLS (e.g. Istio). In Docker Compose, use TLS between containers by mounting certs and configuring each service to use TLS for outbound connections.

## NeuronDesktop API

- Run the API behind a reverse proxy that terminates TLS (recommended). Set `SESSION_SECURE=true` so cookies are only sent over HTTPS.
- Optional: run the API server with TLS by providing certificate and key (implementation may require adding TLS config to the Go server).

## NeuronAgent

- Expose NeuronAgent behind a load balancer or ingress with TLS. Health check and API are then accessed via HTTPS.

## NeuronMCP

- MCP is typically used over stdio or a local TCP connection. For remote MCP, put a TLS-terminating proxy in front.

## NeuronDB (PostgreSQL)

- Configure PostgreSQL with `ssl = on` and provide server certificate and key. Clients use `sslmode=verify-full` or `verify-ca` and CA certificate for verification.
- In production compose, do not expose the PostgreSQL port on the host; only other containers (or internal LB) connect with TLS to the DB.

## Certificate rotation

- Use **cert-manager** in Kubernetes for automatic issuance and renewal (Let's Encrypt or internal CA).
- For Vault PKI, use Vault Agent or cert-manager with Vault issuer to rotate certs and reload services (or restart pods to pick new certs).
