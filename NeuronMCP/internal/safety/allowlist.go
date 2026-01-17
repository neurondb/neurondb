/*-------------------------------------------------------------------------
 *
 * allowlist.go
 *    Statement allowlist for NeuronMCP safety
 *
 * Provides allowlist functionality for DDL and dangerous commands
 * when safety mode is set to "allowlist".
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/safety/allowlist.go
 *
 *-------------------------------------------------------------------------
 */

package safety

import (
	"strings"
)

/* StatementAllowlist manages allowed SQL statements */
type StatementAllowlist struct {
	allowedStatements map[string]bool
	allowedPrefixes   []string
}

/* NewStatementAllowlist creates a new statement allowlist */
func NewStatementAllowlist(statements []string) *StatementAllowlist {
	allowlist := &StatementAllowlist{
		allowedStatements: make(map[string]bool),
		allowedPrefixes:   []string{},
	}

	/* Common safe prefixes */
	safePrefixes := []string{
		"SELECT",
		"WITH", /* CTE queries start with WITH */
	}

	for _, prefix := range safePrefixes {
		allowlist.allowedPrefixes = append(allowlist.allowedPrefixes, prefix)
	}

	/* Add provided statements */
	for _, stmt := range statements {
		stmtUpper := strings.ToUpper(strings.TrimSpace(stmt))
		if stmtUpper != "" {
			/* Check if it's a prefix pattern (ends with %) */
			if strings.HasSuffix(stmtUpper, "%") {
				prefix := strings.TrimSuffix(stmtUpper, "%")
				allowlist.allowedPrefixes = append(allowlist.allowedPrefixes, prefix)
			} else {
				allowlist.allowedStatements[stmtUpper] = true
			}
		}
	}

	return allowlist
}

/* IsAllowed checks if a statement is allowed */
func (sa *StatementAllowlist) IsAllowed(queryUpper string) bool {
	if sa == nil {
		return false
	}

	queryUpper = strings.ToUpper(strings.TrimSpace(queryUpper))
	if queryUpper == "" {
		return false
	}

	/* Check exact match */
	if sa.allowedStatements[queryUpper] {
		return true
	}

	/* Check prefix matches */
	for _, prefix := range sa.allowedPrefixes {
		if strings.HasPrefix(queryUpper, prefix) {
			return true
		}
	}

	return false
}

/* AddStatement adds a statement to the allowlist */
func (sa *StatementAllowlist) AddStatement(statement string) {
	if sa == nil {
		return
	}
	stmtUpper := strings.ToUpper(strings.TrimSpace(statement))
	if stmtUpper != "" {
		sa.allowedStatements[stmtUpper] = true
	}
}

/* AddPrefix adds a prefix to the allowlist */
func (sa *StatementAllowlist) AddPrefix(prefix string) {
	if sa == nil {
		return
	}
	prefixUpper := strings.ToUpper(strings.TrimSpace(prefix))
	if prefixUpper != "" {
		sa.allowedPrefixes = append(sa.allowedPrefixes, prefixUpper)
	}
}

/* GetAllowedStatements returns all allowed statements */
func (sa *StatementAllowlist) GetAllowedStatements() []string {
	if sa == nil {
		return []string{}
	}
	statements := make([]string, 0, len(sa.allowedStatements))
	for stmt := range sa.allowedStatements {
		statements = append(statements, stmt)
	}
	return statements
}

/* GetAllowedPrefixes returns all allowed prefixes */
func (sa *StatementAllowlist) GetAllowedPrefixes() []string {
	if sa == nil {
		return []string{}
	}
	return sa.allowedPrefixes
}



