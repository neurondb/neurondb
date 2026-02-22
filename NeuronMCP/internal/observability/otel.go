/*-------------------------------------------------------------------------
 *
 * otel.go
 *    OpenTelemetry integration for NeuronMCP
 *
 * Provides distributed tracing support with OpenTelemetry.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/observability/otel.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"fmt"
	"sync"
	"time"
)

/* TracerProvider represents a tracing provider with OpenTelemetry support */
/* Note: In production, this would use go.opentelemetry.io/otel */
type TracerProvider struct {
	enabled     bool
	endpoint    string
	serviceName string
	mu          sync.RWMutex
	tracer      *Tracer /* Use existing Tracer */
}

/* NewTracerProvider creates a new tracer provider */
func NewTracerProvider(enabled bool, endpoint, serviceName string) *TracerProvider {
	return &TracerProvider{
		enabled:     enabled,
		endpoint:    endpoint,
		serviceName: serviceName,
		tracer:      NewTracer(),
	}
}

/* StartSpan starts a new span with OpenTelemetry support */
func (tp *TracerProvider) StartSpan(ctx context.Context, name string, opts ...SpanOption) (context.Context, OTELSpan) {
	tp.mu.RLock()
	enabled := tp.enabled
	tracer := tp.tracer
	tp.mu.RUnlock()

	if !enabled || tracer == nil {
		/* Return no-op span */
		return ctx, &NoOpSpan{}
	}

	/* Use existing tracer to create span */
	spanCtx, spanID := tracer.StartSpan(ctx, name)
	span := tracer.GetSpan(spanID)

	/* Apply options */
	for _, opt := range opts {
		opt(&BasicSpan{span: span})
	}

	return spanCtx, &BasicSpan{span: span}
}

/* OTELSpan represents an OpenTelemetry tracing span */
type OTELSpan interface {
	End()
	SetAttribute(key string, value interface{})
	SetStatus(code string, message string)
	RecordError(err error)
}

/* SpanOption configures a span */
type SpanOption func(OTELSpan)

/* BasicSpan is a basic span implementation wrapping the existing Span */
type BasicSpan struct {
	span *Span /* From tracing.go */
}

/* NewBasicSpan creates a new basic span */
func NewBasicSpan(span *Span) *BasicSpan {
	return &BasicSpan{span: span}
}

/* End ends the span */
func (s *BasicSpan) End() {
	if s.span != nil {
		/* End span using existing tracer */
		now := time.Now()
		s.span.EndTime = &now
	}
}

/* SetAttribute sets a span attribute */
func (s *BasicSpan) SetAttribute(key string, value interface{}) {
	if s.span != nil {
		if s.span.Attributes == nil {
			s.span.Attributes = make(map[string]interface{})
		}
		s.span.Attributes[key] = value
	}
}

/* SetStatus sets the span status */
func (s *BasicSpan) SetStatus(code string, message string) {
	if s.span != nil {
		s.span.Status = code
		if s.span.Attributes == nil {
			s.span.Attributes = make(map[string]interface{})
		}
		s.span.Attributes["status.message"] = message
	}
}

/* RecordError records an error in the span */
func (s *BasicSpan) RecordError(err error) {
	if err != nil && s.span != nil {
		s.SetAttribute("error", true)
		s.SetAttribute("error.message", err.Error())
		s.SetStatus("ERROR", err.Error())
	}
}

/* NoOpSpan is a no-op span when tracing is disabled */
type NoOpSpan struct{}

/* End ends the span */
func (s *NoOpSpan) End() {}

/* SetAttribute sets a span attribute */
func (s *NoOpSpan) SetAttribute(key string, value interface{}) {}

/* SetStatus sets the span status */
func (s *NoOpSpan) SetStatus(code string, message string) {}

/* RecordError records an error in the span */
func (s *NoOpSpan) RecordError(err error) {}

/* GetOTELSpanFromContext retrieves an OpenTelemetry span from context */
func GetOTELSpanFromContext(ctx context.Context) OTELSpan {
	if span := GetSpanFromContext(ctx); span != nil {
		return NewBasicSpan(span)
	}
	return &NoOpSpan{}
}

/* TraceToolExecution traces a tool execution */
func TraceToolExecution(ctx context.Context, tracer *TracerProvider, toolName string, fn func(context.Context) error) error {
	spanCtx, span := tracer.StartSpan(ctx, fmt.Sprintf("tool.%s", toolName))
	span.SetAttribute("tool.name", toolName)

	defer span.End()

	err := fn(spanCtx)
	if err != nil {
		span.RecordError(err)
		return err
	}

	span.SetStatus("OK", "")
	return nil
}

/* ExportTrace exports traces to an OTLP collector. */
/* Traces are kept in-memory; this no-op allows callers to enable export later by wiring an OTLP exporter. */
func ExportTrace(ctx context.Context, tracer *TracerProvider, endpoint string) error {
	if !tracer.enabled {
		return nil
	}
	/* In-memory tracing only; wire go.opentelemetry.io/otel/exporters/otlp to export to endpoint */
	return nil
}
