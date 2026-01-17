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

/* SpanKey is the context key for span */
type SpanKey struct{}

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

	/* Try to get trace ID from existing span in context */
	var parentID *SpanID
	if existingSpan, ok := ctx.Value(SpanKey{}).(*Span); ok && existingSpan != nil {
		traceID = existingSpan.TraceID
		existingParentID := existingSpan.SpanID
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

	/* Add span to context - trace ID and span ID are accessible through the span */
	ctx = context.WithValue(ctx, SpanKey{}, span)

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

/* StartSpanWithRequestID starts a new span with request ID from context */
func (t *Tracer) StartSpanWithRequestID(ctx context.Context, name string) (context.Context, SpanID) {
	ctx, spanID := t.StartSpan(ctx, name)
	
	/* Add request ID as span attribute if available */
	/* GetRequestIDFromContext is in the same package (observability) */
	if reqID, ok := GetRequestIDFromContext(ctx); ok {
		t.AddSpanAttribute(spanID, "request_id", reqID.String())
	}
	
	/* Store span in context */
	ctx = context.WithValue(ctx, SpanKey{}, t.GetSpan(spanID))
	
	return ctx, spanID
}

/* GetSpanFromContext retrieves the span from context */
func GetSpanFromContext(ctx context.Context) *Span {
	if ctx == nil {
		return nil
	}
	span, ok := ctx.Value(SpanKey{}).(*Span)
	if !ok {
		return nil
	}
	return span
}

/* AddEvent adds an event to the span */
func (s *Span) AddEvent(name string, attributes map[string]interface{}) {
	if s == nil {
		return
	}
	s.Events = append(s.Events, SpanEvent{
		Name:       name,
		Timestamp:  time.Now(),
		Attributes: attributes,
	})
}

/* SetStatus sets the status of the span */
func (s *Span) SetStatus(status string) {
	if s == nil {
		return
	}
	s.Status = status
}
