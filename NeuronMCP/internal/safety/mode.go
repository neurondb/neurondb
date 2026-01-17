/*-------------------------------------------------------------------------
 *
 * mode.go
 *    Safety mode manager for NeuronMCP
 *
 * Provides safety modes: read-only (default), read-write, and allowlist
 * to control database access and prevent dangerous operations.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/safety/mode.go
 *
 *-------------------------------------------------------------------------
 */

package safety

import (
	"fmt"
	"strings"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* SafetyMode represents the safety mode */
type SafetyMode string

const (
	/* SafetyModeReadOnly is the default mode - only SELECT queries allowed */
	SafetyModeReadOnly SafetyMode = "read_only"
	/* SafetyModeReadWrite allows all operations */
	SafetyModeReadWrite SafetyMode = "read_write"
	/* SafetyModeAllowlist only allows statements in the allowlist */
	SafetyModeAllowlist SafetyMode = "allowlist"
)

/* SafetyManager manages safety modes and statement validation */
type SafetyManager struct {
	mode      SafetyMode
	allowlist *StatementAllowlist
	logger    *logging.Logger
}

/* NewSafetyManager creates a new safety manager */
func NewSafetyManager(mode SafetyMode, allowlist *StatementAllowlist, logger *logging.Logger) *SafetyManager {
	if logger == nil {
		/* Create a no-op logger if none provided */
		logger = &logging.Logger{}
	}
	return &SafetyManager{
		mode:      mode,
		allowlist: allowlist,
		logger:    logger,
	}
}

/* ValidateStatement validates a SQL statement against the safety mode */
func (sm *SafetyManager) ValidateStatement(query string, allowWrite bool) error {
	if sm == nil {
		/* No safety manager means no restrictions */
		return nil
	}

	queryUpper := strings.ToUpper(strings.TrimSpace(query))
	if queryUpper == "" {
		return fmt.Errorf("empty query not allowed")
	}

	switch sm.mode {
	case SafetyModeReadOnly:
		/* In read-only mode, only SELECT queries are allowed */
		if !strings.HasPrefix(queryUpper, "SELECT") {
			/* Allow write if explicitly requested */
			if allowWrite {
				sm.logger.Warn("Write operation allowed with explicit flag in read-only mode", map[string]interface{}{
					"query_preview": getQueryPreview(query),
				})
				return nil
			}
			return fmt.Errorf("read-only mode: only SELECT queries are allowed (use allow_write=true to override)")
		}
		return nil

	case SafetyModeReadWrite:
		/* Read-write mode allows all operations */
		return nil

	case SafetyModeAllowlist:
		/* Allowlist mode checks against allowed statements */
		if sm.allowlist == nil {
			return fmt.Errorf("allowlist mode requires statement allowlist configuration")
		}
		if sm.allowlist.IsAllowed(queryUpper) {
			return nil
		}
		/* Check if explicit write flag is set */
		if allowWrite {
			sm.logger.Warn("Write operation allowed with explicit flag in allowlist mode", map[string]interface{}{
				"query_preview": getQueryPreview(query),
			})
			return nil
		}
		return fmt.Errorf("statement not in allowlist: %s (use allow_write=true to override)", getQueryPreview(query))

	default:
		/* Unknown mode defaults to read-only */
		sm.logger.Warn("Unknown safety mode, defaulting to read-only", map[string]interface{}{
			"mode": sm.mode,
		})
		if !strings.HasPrefix(queryUpper, "SELECT") {
			if allowWrite {
				return nil
			}
			return fmt.Errorf("read-only mode: only SELECT queries are allowed")
		}
		return nil
	}
}

/* GetMode returns the current safety mode */
func (sm *SafetyManager) GetMode() SafetyMode {
	if sm == nil {
		return SafetyModeReadOnly
	}
	return sm.mode
}

/* IsReadOnly returns true if the safety mode is read-only */
func (sm *SafetyManager) IsReadOnly() bool {
	return sm.GetMode() == SafetyModeReadOnly
}

/* getQueryPreview returns a preview of the query for logging */
func getQueryPreview(query string) string {
	previewLen := 100
	if len(query) < previewLen {
		previewLen = len(query)
	}
	if previewLen == 0 {
		return ""
	}
	return query[:previewLen]
}



