/*-------------------------------------------------------------------------
 *
 * request_id.go
 *    Request ID generation for NeuronMCP
 *
 * Provides unique request ID generation for each tool call to enable
 * request tracing and correlation across logs and traces.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/observability/request_id.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"fmt"

	"github.com/google/uuid"
)

/* RequestIDKey is the context key for request ID */
type RequestIDKey struct{}

/* RequestID represents a unique request identifier */
type RequestID string

/* GenerateRequestID generates a new unique request ID */
func GenerateRequestID() RequestID {
	return RequestID(uuid.New().String())
}

/* GetRequestIDFromContext retrieves the request ID from context */
func GetRequestIDFromContext(ctx context.Context) (RequestID, bool) {
	reqID, ok := ctx.Value(RequestIDKey{}).(RequestID)
	return reqID, ok
}

/* WithRequestID adds a request ID to the context */
func WithRequestID(ctx context.Context, reqID RequestID) context.Context {
	return context.WithValue(ctx, RequestIDKey{}, reqID)
}

/* GetOrCreateRequestID gets an existing request ID from context or creates a new one */
func GetOrCreateRequestID(ctx context.Context) (context.Context, RequestID) {
	if reqID, ok := GetRequestIDFromContext(ctx); ok {
		return ctx, reqID
	}
	reqID := GenerateRequestID()
	return WithRequestID(ctx, reqID), reqID
}

/* String returns the string representation of the request ID */
func (r RequestID) String() string {
	return string(r)
}

/* IsValid checks if the request ID is a valid UUID */
func (r RequestID) IsValid() bool {
	_, err := uuid.Parse(string(r))
	return err == nil
}

/* Format formats the request ID for logging */
func (r RequestID) Format() string {
	return fmt.Sprintf("req_id=%s", r.String())
}



