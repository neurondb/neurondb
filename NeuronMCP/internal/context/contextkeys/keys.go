/*-------------------------------------------------------------------------
 *
 * keys.go
 *    Typed context keys for NeuronMCP
 *
 * Provides typed context keys to prevent key collisions when using
 * context.WithValue. All context keys should be defined here as empty
 * struct types (following Go best practices).
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/context/contextkeys
 *
 *-------------------------------------------------------------------------
 */

package contextkeys

/* Authentication keys */
type UserKey struct{}
type UserIDKey struct{}

/* Tenant/Organization keys */
type OrgIDKey struct{}
type ProjectIDKey struct{}

/* Authorization keys */
type ScopesKey struct{}

/* Observability keys (TraceID and SpanID are handled in observability package) */
/* RequestIDKey is already defined in observability/request_id.go */

/* Audit keys */
type AuditContextKey struct{}

/* HTTP metadata key */
type HTTPMetadataKey struct{}
