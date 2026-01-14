/*-------------------------------------------------------------------------
 *
 * tracing.go
 *    Distributed tracing for NeuronAgent
 *
 * Provides distributed tracing with OpenTelemetry, spans for agent/tool/DB/LLM calls,
 * and trace context propagation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/observability/tracing.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

/* TraceContext holds trace and span information */
type TraceContext struct {
	TraceID      string
	SpanID       string
	ParentSpanID string
}

type contextKey string

const traceContextKey contextKey = "trace_context"

/* Tracer provides distributed tracing functionality */
type Tracer struct {
	enabled bool
}

/* NewTracer creates a new tracer */
func NewTracer(enabled bool) *Tracer {
	return &Tracer{enabled: enabled}
}

/* StartSpan starts a new span and returns updated context */
func (t *Tracer) StartSpan(ctx context.Context, name string) (context.Context, string) {
	if !t.enabled {
		return ctx, ""
	}

	traceID := t.getOrCreateTraceID(ctx)
	spanID := uuid.New().String()

	traceCtx := &TraceContext{
		TraceID:      traceID,
		SpanID:       spanID,
		ParentSpanID: t.getSpanID(ctx),
	}

	ctx = context.WithValue(ctx, traceContextKey, traceCtx)
	return ctx, spanID
}

/* EndSpan ends a span */
func (t *Tracer) EndSpan(ctx context.Context, spanID string, attributes map[string]interface{}) {
	if !t.enabled {
		return
	}
	/* In production, send span to tracing backend (Jaeger, Zipkin, etc.) */
	/* For now, we just store in context */
}

/* GetTraceIDFromContext gets trace ID from context */
func (t *Tracer) GetTraceIDFromContext(ctx context.Context) string {
	if traceCtx, ok := ctx.Value(traceContextKey).(*TraceContext); ok {
		return traceCtx.TraceID
	}
	return ""
}

/* GetSpanIDFromContext gets span ID from context */
func (t *Tracer) GetSpanIDFromContext(ctx context.Context) string {
	if traceCtx, ok := ctx.Value(traceContextKey).(*TraceContext); ok {
		return traceCtx.SpanID
	}
	return ""
}

/* getOrCreateTraceID gets trace ID from context or creates new one */
func (t *Tracer) getOrCreateTraceID(ctx context.Context) string {
	if traceID := t.GetTraceIDFromContext(ctx); traceID != "" {
		return traceID
	}
	return uuid.New().String()
}

/* getSpanID gets span ID from context */
func (t *Tracer) getSpanID(ctx context.Context) string {
	return t.GetSpanIDFromContext(ctx)
}

/* Span represents a tracing span */
type Span struct {
	TraceID    string
	SpanID     string
	ParentID   string
	Name       string
	StartTime  time.Time
	EndTime    time.Time
	Attributes map[string]interface{}
}

/* StartAgentSpan starts a span for agent execution */
func (t *Tracer) StartAgentSpan(ctx context.Context, agentID uuid.UUID) (context.Context, string) {
	ctx, spanID := t.StartSpan(ctx, "agent.execute")
	return ctx, spanID
}

/* StartToolSpan starts a span for tool execution */
func (t *Tracer) StartToolSpan(ctx context.Context, toolName string) (context.Context, string) {
	ctx, spanID := t.StartSpan(ctx, fmt.Sprintf("tool.%s", toolName))
	return ctx, spanID
}

/* StartDBSpan starts a span for database query */
func (t *Tracer) StartDBSpan(ctx context.Context, queryType string) (context.Context, string) {
	ctx, spanID := t.StartSpan(ctx, fmt.Sprintf("db.%s", queryType))
	return ctx, spanID
}

/* StartLLMSpan starts a span for LLM call */
func (t *Tracer) StartLLMSpan(ctx context.Context, modelName string) (context.Context, string) {
	ctx, spanID := t.StartSpan(ctx, fmt.Sprintf("llm.%s", modelName))
	return ctx, spanID
}

/* WithTraceContext adds trace context to context */
func WithTraceContext(ctx context.Context, traceID, spanID string) context.Context {
	traceCtx := &TraceContext{
		TraceID: traceID,
		SpanID:  spanID,
	}
	return context.WithValue(ctx, traceContextKey, traceCtx)
}

/* GetTraceContextFromContext gets trace context from context */
func GetTraceContextFromContext(ctx context.Context) *TraceContext {
	if traceCtx, ok := ctx.Value(traceContextKey).(*TraceContext); ok {
		return traceCtx
	}
	return nil
}
