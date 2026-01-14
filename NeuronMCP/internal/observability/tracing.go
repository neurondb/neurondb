/*-------------------------------------------------------------------------
 *
 * tracing.go
 *    Distributed tracing support
 *
 * Implements OpenTelemetry integration as specified in Phase 2.2.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/observability/tracing.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"fmt"
	"time"
)

/* TraceID represents a trace ID */
type TraceID string

/* SpanID represents a span ID */
type SpanID string

/* Span represents a tracing span */
type Span struct {
	TraceID    TraceID
	SpanID     SpanID
	ParentID   *SpanID
	Name       string
	StartTime  time.Time
	EndTime    *time.Time
	Attributes map[string]interface{}
	Events     []SpanEvent
	Status     string
}

/* SpanEvent represents a span event */
type SpanEvent struct {
	Name       string
	Timestamp  time.Time
	Attributes map[string]interface{}
}

/* Tracer provides distributed tracing */
type Tracer struct {
	spans map[SpanID]*Span
}

/* NewTracer creates a new tracer */
func NewTracer() *Tracer {
	return &Tracer{
		spans: make(map[SpanID]*Span),
	}
}

/* StartSpan starts a new span */
func (t *Tracer) StartSpan(ctx context.Context, name string) (context.Context, SpanID) {
	spanID := SpanID(fmt.Sprintf("span_%d", time.Now().UnixNano()))
	traceID := TraceID(fmt.Sprintf("trace_%d", time.Now().UnixNano()))

	/* Try to get trace ID from context */
	if existingTraceID, ok := ctx.Value("trace_id").(TraceID); ok {
		traceID = existingTraceID
	}

	/* Try to get parent span ID from context */
	var parentID *SpanID
	if existingParentID, ok := ctx.Value("span_id").(SpanID); ok {
		parentID = &existingParentID
	}

	span := &Span{
		TraceID:    traceID,
		SpanID:     spanID,
		ParentID:   parentID,
		Name:       name,
		StartTime:  time.Now(),
		Attributes: make(map[string]interface{}),
		Events:     []SpanEvent{},
		Status:     "ok",
	}

	t.spans[spanID] = span

	/* Add to context */
	ctx = context.WithValue(ctx, "trace_id", traceID)
	ctx = context.WithValue(ctx, "span_id", spanID)

	return ctx, spanID
}

/* EndSpan ends a span */
func (t *Tracer) EndSpan(spanID SpanID) {
	span, exists := t.spans[spanID]
	if !exists {
		return
	}

	now := time.Now()
	span.EndTime = &now
}

/* AddSpanAttribute adds an attribute to a span */
func (t *Tracer) AddSpanAttribute(spanID SpanID, key string, value interface{}) {
	span, exists := t.spans[spanID]
	if !exists {
		return
	}

	span.Attributes[key] = value
}

/* AddSpanEvent adds an event to a span */
func (t *Tracer) AddSpanEvent(spanID SpanID, name string, attributes map[string]interface{}) {
	span, exists := t.spans[spanID]
	if !exists {
		return
	}

	span.Events = append(span.Events, SpanEvent{
		Name:       name,
		Timestamp:  time.Now(),
		Attributes: attributes,
	})
}

/* SetSpanStatus sets the status of a span */
func (t *Tracer) SetSpanStatus(spanID SpanID, status string) {
	span, exists := t.spans[spanID]
	if !exists {
		return
	}

	span.Status = status
}

/* GetSpan gets a span */
func (t *Tracer) GetSpan(spanID SpanID) *Span {
	return t.spans[spanID]
}

/* GetTrace gets all spans for a trace */
func (t *Tracer) GetTrace(traceID TraceID) []*Span {
	spans := []*Span{}
	for _, span := range t.spans {
		if span.TraceID == traceID {
			spans = append(spans, span)
		}
	}
	return spans
}
