/*-------------------------------------------------------------------------
 *
 * errors.go
 *    Error taxonomy for NeuronMCP
 *
 * Provides structured error types and taxonomy to replace raw driver errors
 * with clear, actionable error messages.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/reliability/errors.go
 *
 *-------------------------------------------------------------------------
 */

package reliability

import (
	"strings"
)

/* ErrorCode represents a structured error code */
type ErrorCode string

const (
	/* ErrorCodeTimeout indicates a timeout error */
	ErrorCodeTimeout ErrorCode = "TIMEOUT"
	/* ErrorCodeConnection indicates a connection error */
	ErrorCodeConnection ErrorCode = "CONNECTION_ERROR"
	/* ErrorCodeValidation indicates a validation error */
	ErrorCodeValidation ErrorCode = "VALIDATION_ERROR"
	/* ErrorCodePermission indicates a permission error */
	ErrorCodePermission ErrorCode = "PERMISSION_DENIED"
	/* ErrorCodeQuery indicates a query execution error */
	ErrorCodeQuery ErrorCode = "QUERY_ERROR"
	/* ErrorCodeSafety indicates a safety violation */
	ErrorCodeSafety ErrorCode = "SAFETY_VIOLATION"
	/* ErrorCodeReadOnlyViolation indicates a read-only mode violation */
	ErrorCodeReadOnlyViolation ErrorCode = "READ_ONLY_VIOLATION"
	/* ErrorCodeNotFound indicates a resource not found */
	ErrorCodeNotFound ErrorCode = "NOT_FOUND"
	/* ErrorCodeInvalidParameter indicates an invalid parameter */
	ErrorCodeInvalidParameter ErrorCode = "INVALID_PARAMETER"
	/* ErrorCodeInternal indicates an internal server error */
	ErrorCodeInternal ErrorCode = "INTERNAL_ERROR"
	/* ErrorCodeRetryExhausted indicates retries were exhausted */
	ErrorCodeRetryExhausted ErrorCode = "RETRY_EXHAUSTED"
	/* ErrorCodeConfirmationRequired indicates confirmation is required */
	ErrorCodeConfirmationRequired ErrorCode = "CONFIRMATION_REQUIRED"
)

/* StructuredError represents a structured error with code and details */
type StructuredError struct {
	Code         ErrorCode
	Message      string
	Details      map[string]interface{}
	RequestID    string
	OriginalError error
	Suggestions  []string /* Helpful suggestions for resolving the error */
}

/* Error returns the error message */
func (e *StructuredError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	if e.OriginalError != nil {
		return e.OriginalError.Error()
	}
	return string(e.Code)
}

/* Unwrap returns the original error */
func (e *StructuredError) Unwrap() error {
	return e.OriginalError
}

/* NewStructuredError creates a new structured error */
func NewStructuredError(code ErrorCode, message string, details map[string]interface{}) *StructuredError {
	return &StructuredError{
		Code:    code,
		Message: message,
		Details: details,
	}
}

/* WithRequestID adds a request ID to the error */
func (e *StructuredError) WithRequestID(requestID string) *StructuredError {
	e.RequestID = requestID
	if e.Details == nil {
		e.Details = make(map[string]interface{})
	}
	e.Details["request_id"] = requestID
	return e
}

/* WithOriginalError adds the original error */
func (e *StructuredError) WithOriginalError(err error) *StructuredError {
	e.OriginalError = err
	return e
}

/* WithSuggestions adds helpful suggestions */
func (e *StructuredError) WithSuggestions(suggestions ...string) *StructuredError {
	e.Suggestions = suggestions
	if e.Details == nil {
		e.Details = make(map[string]interface{})
	}
	e.Details["suggestions"] = suggestions
	return e
}

/* GetSuggestions returns error-specific suggestions */
func GetSuggestions(err error) []string {
	if structuredErr, ok := err.(*StructuredError); ok {
		if len(structuredErr.Suggestions) > 0 {
			return structuredErr.Suggestions
		}
		/* Generate suggestions based on error code */
		return generateSuggestions(structuredErr.Code, structuredErr.Details)
	}
	return nil
}

/* generateSuggestions generates suggestions based on error code */
func generateSuggestions(code ErrorCode, details map[string]interface{}) []string {
	suggestions := []string{}
	
	switch code {
	case ErrorCodeConnection:
		suggestions = append(suggestions,
			"Check if the database server is running",
			"Verify connection parameters (host, port, database, user)",
			"Check network connectivity and firewall settings",
			"Ensure the database user has proper permissions",
		)
	case ErrorCodeTimeout:
		suggestions = append(suggestions,
			"Increase the timeout value if the operation is expected to take longer",
			"Check database performance and query optimization",
			"Consider breaking large operations into smaller batches",
		)
	case ErrorCodeValidation:
		suggestions = append(suggestions,
			"Review the request parameters and ensure all required fields are provided",
			"Check parameter types and value ranges",
			"Refer to the tool documentation for parameter requirements",
		)
	case ErrorCodePermission:
		suggestions = append(suggestions,
			"Verify that the database user has the required permissions",
			"Check if the operation is allowed in the current safety mode",
			"Contact your database administrator for access",
		)
	case ErrorCodeQuery:
		suggestions = append(suggestions,
			"Review the SQL query syntax",
			"Check if referenced tables, columns, or functions exist",
			"Verify data types match expected values",
		)
	case ErrorCodeNotFound:
		suggestions = append(suggestions,
			"Verify that the resource exists",
			"Check the resource name or identifier",
			"List available resources to see what's available",
		)
	case ErrorCodeReadOnlyViolation:
		suggestions = append(suggestions,
			"This operation requires write access",
			"Check if the server is in read-only mode",
			"Enable write access in the configuration if needed",
		)
	}
	
	return suggestions
}

/* ErrorClassifier classifies errors into taxonomy */
type ErrorClassifier struct {
}

/* NewErrorClassifier creates a new error classifier */
func NewErrorClassifier() *ErrorClassifier {
	return &ErrorClassifier{}
}

/* ClassifyError classifies an error into the taxonomy */
func (ec *ErrorClassifier) ClassifyError(err error, requestID string) *StructuredError {
	if err == nil {
		return nil
	}

	errorStr := strings.ToLower(err.Error())

	/* Check for timeout errors */
	if strings.Contains(errorStr, "timeout") || strings.Contains(errorStr, "deadline exceeded") {
		return NewStructuredError(ErrorCodeTimeout, "Operation timed out", nil).
			WithRequestID(requestID).
			WithOriginalError(err).
			WithSuggestions(
				"Increase the timeout value if the operation is expected to take longer",
				"Check database performance and query optimization",
				"Consider breaking large operations into smaller batches",
			)
	}

	/* Check for connection errors */
	if strings.Contains(errorStr, "connection") || strings.Contains(errorStr, "network") ||
		strings.Contains(errorStr, "unreachable") || strings.Contains(errorStr, "refused") {
		return NewStructuredError(ErrorCodeConnection, "Database connection error", nil).
			WithRequestID(requestID).
			WithOriginalError(err).
			WithSuggestions(
				"Check if the database server is running",
				"Verify connection parameters (host, port, database, user)",
				"Check network connectivity and firewall settings",
			)
	}

	/* Check for permission errors */
	if strings.Contains(errorStr, "permission") || strings.Contains(errorStr, "denied") ||
		strings.Contains(errorStr, "access") || strings.Contains(errorStr, "unauthorized") {
		return NewStructuredError(ErrorCodePermission, "Permission denied", nil).
			WithRequestID(requestID).
			WithOriginalError(err)
	}

	/* Check for validation errors */
	if strings.Contains(errorStr, "validation") || strings.Contains(errorStr, "invalid") ||
		strings.Contains(errorStr, "required") {
		return NewStructuredError(ErrorCodeValidation, "Validation error", nil).
			WithRequestID(requestID).
			WithOriginalError(err)
	}

	/* Check for safety violations */
	if strings.Contains(errorStr, "safety") || strings.Contains(errorStr, "read-only") ||
		strings.Contains(errorStr, "allowlist") {
		return NewStructuredError(ErrorCodeSafety, "Safety violation", nil).
			WithRequestID(requestID).
			WithOriginalError(err)
	}

	/* Check for not found errors */
	if strings.Contains(errorStr, "not found") || strings.Contains(errorStr, "does not exist") {
		return NewStructuredError(ErrorCodeNotFound, "Resource not found", nil).
			WithRequestID(requestID).
			WithOriginalError(err)
	}

	/* Check for query errors (PostgreSQL specific) */
	if strings.Contains(errorStr, "syntax error") || strings.Contains(errorStr, "sql") ||
		strings.Contains(errorStr, "query") {
		/* Extract a cleaner message from PostgreSQL errors */
		message := "Query execution failed"
		if idx := strings.Index(errorStr, "error:"); idx >= 0 {
			message = strings.TrimSpace(errorStr[idx+6:])
			if len(message) > 200 {
				message = message[:200] + "..."
			}
		}
		return NewStructuredError(ErrorCodeQuery, message, nil).
			WithRequestID(requestID).
			WithOriginalError(err)
	}

	/* Default: internal error (don't expose raw error) */
	return NewStructuredError(ErrorCodeInternal, "An internal error occurred", map[string]interface{}{
		"error_type": "internal",
	}).
		WithRequestID(requestID).
		WithOriginalError(err)
}

/* ClassifyPostgresError classifies a PostgreSQL-specific error */
func (ec *ErrorClassifier) ClassifyPostgresError(err error, requestID string) *StructuredError {
	if err == nil {
		return nil
	}

	errorStr := err.Error()

	/* PostgreSQL error codes (common ones) */
	pgErrorPatterns := map[string]ErrorCode{
		"42P01": ErrorCodeNotFound,        /* undefined_table */
		"42703": ErrorCodeQuery,          /* undefined_column */
		"23505": ErrorCodeQuery,          /* unique_violation */
		"23503": ErrorCodeQuery,          /* foreign_key_violation */
		"28P01": ErrorCodePermission,     /* invalid_authorization_specification */
		"3D000": ErrorCodeConnection,      /* invalid_catalog_name */
		"08003": ErrorCodeConnection,      /* connection_does_not_exist */
		"08006": ErrorCodeConnection,      /* connection_failure */
		"57014": ErrorCodeTimeout,        /* query_canceled */
		"53300": ErrorCodeConnection,      /* too_many_connections */
	}

	/* Check for PostgreSQL error codes */
	for code, errorCode := range pgErrorPatterns {
		if strings.Contains(errorStr, code) {
			return NewStructuredError(errorCode, ec.getPostgresErrorMessage(errorStr), nil).
				WithRequestID(requestID).
				WithOriginalError(err)
		}
	}

	/* Fall back to general classification */
	return ec.ClassifyError(err, requestID)
}

/* getPostgresErrorMessage extracts a user-friendly message from PostgreSQL error */
func (ec *ErrorClassifier) getPostgresErrorMessage(errorStr string) string {
	/* Try to extract the actual error message */
	if idx := strings.Index(errorStr, "ERROR:"); idx >= 0 {
		msg := strings.TrimSpace(errorStr[idx+6:])
		/* Remove error code if present */
		if idx2 := strings.Index(msg, ":"); idx2 >= 0 {
			msg = strings.TrimSpace(msg[idx2+1:])
		}
		if len(msg) > 200 {
			msg = msg[:200] + "..."
		}
		return msg
	}
	return "Database operation failed"
}

/* FormatError formats an error for user display */
func FormatError(err error, requestID string) string {
	if err == nil {
		return ""
	}

	/* Check if it's already a structured error */
	if structuredErr, ok := err.(*StructuredError); ok {
		return structuredErr.Message
	}

	/* Classify the error */
	classifier := NewErrorClassifier()
	structuredErr := classifier.ClassifyError(err, requestID)
	return structuredErr.Message
}

/* GetErrorCode extracts the error code from an error */
func GetErrorCode(err error) ErrorCode {
	if structuredErr, ok := err.(*StructuredError); ok {
		return structuredErr.Code
	}
	return ErrorCodeInternal
}

/* IsRetryableError checks if an error is retryable */
func IsRetryableError(err error) bool {
	code := GetErrorCode(err)
	retryableCodes := []ErrorCode{
		ErrorCodeTimeout,
		ErrorCodeConnection,
	}
	for _, retryableCode := range retryableCodes {
		if code == retryableCode {
			return true
		}
	}
	return false
}

