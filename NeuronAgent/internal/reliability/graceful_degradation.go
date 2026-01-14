/*-------------------------------------------------------------------------
 *
 * graceful_degradation.go
 *    Graceful degradation when services are unavailable
 *
 * Provides fallback mechanisms and degraded modes when services
 * fail or are unavailable.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/reliability/graceful_degradation.go
 *
 *-------------------------------------------------------------------------
 */

package reliability

import (
	"context"
	"fmt"
)

/* DegradationMode represents the level of service degradation */
type DegradationMode string

const (
	ModeFull     DegradationMode = "full"     /* Full functionality */
	ModeDegraded DegradationMode = "degraded" /* Reduced functionality */
	ModeMinimal  DegradationMode = "minimal"  /* Minimal functionality */
)

/* GracefulDegradation manages graceful degradation */
type GracefulDegradation struct {
	mode           DegradationMode
	fallbackFunc   func(context.Context) (interface{}, error)
	degredationFunc func(context.Context) (interface{}, error)
}

/* NewGracefulDegradation creates graceful degradation manager */
func NewGracefulDegradation(fallback, degraded func(context.Context) (interface{}, error)) *GracefulDegradation {
	return &GracefulDegradation{
		mode:            ModeFull,
		fallbackFunc:    fallback,
		degredationFunc: degraded,
	}
}

/* ExecuteWithFallback executes function with fallback */
func (gd *GracefulDegradation) ExecuteWithFallback(ctx context.Context, primary func(context.Context) (interface{}, error)) (interface{}, error) {
	/* Try primary function */
	result, err := primary(ctx)
	if err == nil {
		return result, nil
	}

	/* Try fallback */
	if gd.fallbackFunc != nil {
		result, err := gd.fallbackFunc(ctx)
		if err == nil {
			return result, nil
		}
	}

	/* Try degraded mode */
	if gd.degredationFunc != nil {
		result, err := gd.degredationFunc(ctx)
		if err == nil {
			return result, nil
		}
	}

	return nil, fmt.Errorf("all service modes failed: primary_error=%w", err)
}

/* SetMode sets degradation mode */
func (gd *GracefulDegradation) SetMode(mode DegradationMode) {
	gd.mode = mode
}

/* GetMode returns current degradation mode */
func (gd *GracefulDegradation) GetMode() DegradationMode {
	return gd.mode
}



