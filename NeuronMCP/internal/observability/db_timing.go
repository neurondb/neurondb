/*-------------------------------------------------------------------------
 *
 * db_timing.go
 *    Database timing tracker for NeuronMCP
 *
 * Tracks database query execution times and records slow queries
 * for observability and performance monitoring.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/observability/db_timing.go
 *
 *-------------------------------------------------------------------------
 */

package observability

import (
	"context"
	"time"
)

/* DBTimingTracker tracks database query execution times */
type DBTimingTracker struct {
	slowQueryThreshold time.Duration
}

/* NewDBTimingTracker creates a new DB timing tracker */
func NewDBTimingTracker(slowQueryThreshold time.Duration) *DBTimingTracker {
	if slowQueryThreshold == 0 {
		slowQueryThreshold = 1 * time.Second /* Default: 1 second */
	}
	return &DBTimingTracker{
		slowQueryThreshold: slowQueryThreshold,
	}
}

/* TrackQuery tracks a database query execution */
func (t *DBTimingTracker) TrackQuery(ctx context.Context, query string, duration time.Duration) {
	if t == nil {
		return
	}

	/* Record query timing as span event if tracing is enabled */
	if span := GetSpanFromContext(ctx); span != nil {
		span.AddEvent("db.query", map[string]interface{}{
			"query":    getQueryPreview(query),
			"duration": duration.Milliseconds(),
			"slow":     duration > t.slowQueryThreshold,
		})
	}

	/* Mark as slow query if exceeds threshold */
	if duration > t.slowQueryThreshold {
		/* This would typically be logged or sent to metrics */
		/* For now, we rely on span events */
	}
}

/* TrackQueryWithResult tracks a database query with result information */
func (t *DBTimingTracker) TrackQueryWithResult(ctx context.Context, query string, duration time.Duration, rowsAffected int, err error) {
	if t == nil {
		return
	}

	t.TrackQuery(ctx, query, duration)

	if span := GetSpanFromContext(ctx); span != nil {
		attributes := map[string]interface{}{
			"rows_affected": rowsAffected,
		}
		if err != nil {
			attributes["error"] = err.Error()
			if span != nil { span.SetStatus("error") }
		} else {
			if span != nil { span.SetStatus("ok") }
		}
		if span != nil { span.AddEvent("db.query.result", attributes) }
	}
}

/* getQueryPreview returns a preview of the query */
func getQueryPreview(query string) string {
	previewLen := 200
	if len(query) < previewLen {
		previewLen = len(query)
	}
	if previewLen == 0 {
		return ""
	}
	return query[:previewLen]
}



