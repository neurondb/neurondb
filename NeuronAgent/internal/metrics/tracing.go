/*-------------------------------------------------------------------------
 *
 * tracing.go
 *    Distributed tracing support for NeuronAgent
 *
 * Provides trace and span management for tracking request flows across
 * services with trace ID generation and span context propagation.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/metrics/tracing.go
 *
 *-------------------------------------------------------------------------
 */

package metrics

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

type TraceID string

type Span struct {
	TraceID    TraceID
	SpanID     string
	ParentID   string
	Name       string
	StartTime  time.Time
	EndTime    time.Time
	Attributes map[string]interface{}
}

type Tracer struct {
	spans map[string]*Span
}

func NewTracer() *Tracer {
	return &Tracer{
		spans: make(map[string]*Span),
	}
}

/* StartSpan starts a new span */
func (t *Tracer) StartSpan(ctx context.Context, name string) (context.Context, string) {
	spanID := uuid.New().String()
	traceID := TraceID(uuid.New().String())

	/* Try to get trace ID from context */
	if existingTraceID := ctx.Value("trace_id"); existingTraceID != nil {
		traceID = existingTraceID.(TraceID)
	}

	span := &Span{
		TraceID:    traceID,
		SpanID:     spanID,
		Name:       name,
		StartTime:  time.Now(),
		Attributes: make(map[string]interface{}),
	}

	/* Get parent span ID if exists */
	if parentSpanID := ctx.Value("span_id"); parentSpanID != nil {
		span.ParentID = parentSpanID.(string)
	}

	t.spans[spanID] = span

	/* Add to context */
	ctx = context.WithValue(ctx, "trace_id", traceID)
	ctx = context.WithValue(ctx, "span_id", spanID)

	return ctx, spanID
}

/* EndSpan ends a span */
func (t *Tracer) EndSpan(spanID string) {
	if span, exists := t.spans[spanID]; exists {
		span.EndTime = time.Now()
		/* In production, send to tracing backend */
		delete(t.spans, spanID)
	}
}

/* AddAttribute adds an attribute to a span */
func (t *Tracer) AddAttribute(spanID string, key string, value interface{}) {
	if span, exists := t.spans[spanID]; exists {
		span.Attributes[key] = value
	}
}

/* GetSpan returns a span by ID */
func (t *Tracer) GetSpan(spanID string) (*Span, error) {
	span, exists := t.spans[spanID]
	if !exists {
		return nil, fmt.Errorf("span not found: %s", spanID)
	}
	return span, nil
}

/* GetTraceIDFromContextTyped gets trace ID from context as TraceID type */
func GetTraceIDFromContextTyped(ctx context.Context) (TraceID, bool) {
	traceID, ok := ctx.Value("trace_id").(TraceID)
	return traceID, ok
}

/* GetSpanIDFromContext gets span ID from context */
func GetSpanIDFromContext(ctx context.Context) (string, bool) {
	spanID, ok := ctx.Value("span_id").(string)
	return spanID, ok
}

/* FormatTraceHeader formats trace ID for HTTP headers */
func (t TraceID) String() string {
	return string(t)
}
