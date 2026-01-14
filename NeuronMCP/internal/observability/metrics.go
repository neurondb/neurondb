/*-------------------------------------------------------------------------
 *
 * metrics.go
 *    Comprehensive metrics collection for NeuronMCP
 *
 * Implements metrics expansion as specified in Phase 2.2.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/observability/metrics.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"sync"
	"time"
)

/* MetricType represents a metric type */
type MetricType string

const (
	MetricTypeCounter MetricType = "counter"
	MetricTypeGauge   MetricType = "gauge"
	MetricTypeHistogram MetricType = "histogram"
	MetricTypeSummary   MetricType = "summary"
)

/* Metric represents a metric */
type Metric struct {
	Name      string
	Type      MetricType
	Value     float64
	Labels    map[string]string
	Timestamp time.Time
}

/* MetricsCollector collects metrics */
type MetricsCollector struct {
	metrics map[string]*Metric
	mu      sync.RWMutex
}

/* NewMetricsCollector creates a new metrics collector */
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics: make(map[string]*Metric),
	}
}

/* IncrementCounter increments a counter metric */
func (m *MetricsCollector) IncrementCounter(name string, labels map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := m.makeKey(name, labels)
	if metric, exists := m.metrics[key]; exists {
		metric.Value++
		metric.Timestamp = time.Now()
	} else {
		m.metrics[key] = &Metric{
			Name:      name,
			Type:      MetricTypeCounter,
			Value:     1,
			Labels:    labels,
			Timestamp: time.Now(),
		}
	}
}

/* SetGauge sets a gauge metric */
func (m *MetricsCollector) SetGauge(name string, value float64, labels map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := m.makeKey(name, labels)
	m.metrics[key] = &Metric{
		Name:      name,
		Type:      MetricTypeGauge,
		Value:     value,
		Labels:    labels,
		Timestamp: time.Now(),
	}
}

/* ObserveHistogram observes a histogram value */
func (m *MetricsCollector) ObserveHistogram(name string, value float64, labels map[string]string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := m.makeKey(name, labels)
	if metric, exists := m.metrics[key]; exists {
		/* In production, maintain histogram buckets */
		metric.Value = value
		metric.Timestamp = time.Now()
	} else {
		m.metrics[key] = &Metric{
			Name:      name,
			Type:      MetricTypeHistogram,
			Value:     value,
			Labels:    labels,
			Timestamp: time.Now(),
		}
	}
}

/* GetMetric gets a metric */
func (m *MetricsCollector) GetMetric(name string, labels map[string]string) *Metric {
	m.mu.RLock()
	defer m.mu.RUnlock()

	key := m.makeKey(name, labels)
	return m.metrics[key]
}

/* GetAllMetrics returns all metrics */
func (m *MetricsCollector) GetAllMetrics() map[string]*Metric {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*Metric)
	for k, v := range m.metrics {
		result[k] = v
	}
	return result
}

/* makeKey creates a key from name and labels */
func (m *MetricsCollector) makeKey(name string, labels map[string]string) string {
	key := name
	for k, v := range labels {
		key += ":" + k + "=" + v
	}
	return key
}

/* BusinessMetrics tracks business metrics */
type BusinessMetrics struct {
	*MetricsCollector
}

/* NewBusinessMetrics creates business metrics */
func NewBusinessMetrics() *BusinessMetrics {
	return &BusinessMetrics{
		MetricsCollector: NewMetricsCollector(),
	}
}

/* RecordToolUsage records tool usage */
func (b *BusinessMetrics) RecordToolUsage(toolName, userID string) {
	b.IncrementCounter("tool_usage_total", map[string]string{
		"tool": toolName,
		"user": userID,
	})
}

/* RecordUserActivity records user activity */
func (b *BusinessMetrics) RecordUserActivity(userID, activity string) {
	b.IncrementCounter("user_activity_total", map[string]string{
		"user":     userID,
		"activity": activity,
	})
}

/* PerformanceMetrics tracks performance metrics */
type PerformanceMetrics struct {
	*MetricsCollector
}

/* NewPerformanceMetrics creates performance metrics */
func NewPerformanceMetrics() *PerformanceMetrics {
	return &PerformanceMetrics{
		MetricsCollector: NewMetricsCollector(),
	}
}

/* RecordLatency records latency */
func (p *PerformanceMetrics) RecordLatency(operation string, duration time.Duration) {
	p.ObserveHistogram("operation_latency_seconds", duration.Seconds(), map[string]string{
		"operation": operation,
	})
}

/* RecordThroughput records throughput */
func (p *PerformanceMetrics) RecordThroughput(operation string, count int) {
	p.IncrementCounter("operation_throughput_total", map[string]string{
		"operation": operation,
	})
}

/* ErrorMetrics tracks error metrics */
type ErrorMetrics struct {
	*MetricsCollector
}

/* NewErrorMetrics creates error metrics */
func NewErrorMetrics() *ErrorMetrics {
	return &ErrorMetrics{
		MetricsCollector: NewMetricsCollector(),
	}
}

/* RecordError records an error */
func (e *ErrorMetrics) RecordError(errorType, operation string) {
	e.IncrementCounter("errors_total", map[string]string{
		"type":      errorType,
		"operation": operation,
	})
}

/* ResourceMetrics tracks resource metrics */
type ResourceMetrics struct {
	*MetricsCollector
}

/* NewResourceMetrics creates resource metrics */
func NewResourceMetrics() *ResourceMetrics {
	return &ResourceMetrics{
		MetricsCollector: NewMetricsCollector(),
	}
}

/* RecordCPUUsage records CPU usage */
func (r *ResourceMetrics) RecordCPUUsage(usage float64) {
	r.SetGauge("cpu_usage_percent", usage, nil)
}

/* RecordMemoryUsage records memory usage */
func (r *ResourceMetrics) RecordMemoryUsage(bytes int64) {
	r.SetGauge("memory_usage_bytes", float64(bytes), nil)
}

/* RecordConnections records connection count */
func (r *ResourceMetrics) RecordConnections(count int) {
	r.SetGauge("connections_total", float64(count), nil)
}



