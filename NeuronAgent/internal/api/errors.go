package api

import (
	"fmt"
	"net/http"
)

type APIError struct {
	Code         int
	Message      string
	Err          error
	RequestID    string
	Endpoint     string
	Method       string
	ResourceType string
	ResourceID   string
	Details      map[string]interface{}
}

func (e *APIError) Error() string {
	parts := []string{e.Message}
	
	if e.Endpoint != "" {
		parts = append(parts, fmt.Sprintf("endpoint='%s'", e.Endpoint))
	}
	if e.Method != "" {
		parts = append(parts, fmt.Sprintf("method='%s'", e.Method))
	}
	if e.RequestID != "" {
		parts = append(parts, fmt.Sprintf("request_id='%s'", e.RequestID))
	}
	if e.ResourceType != "" {
		part := fmt.Sprintf("resource_type='%s'", e.ResourceType)
		if e.ResourceID != "" {
			part += fmt.Sprintf(", resource_id='%s'", e.ResourceID)
		}
		parts = append(parts, part)
	}
	
	if e.Err != nil {
		parts = append(parts, fmt.Sprintf("error=%v", e.Err))
	}
	
	return fmt.Sprintf("%s", fmt.Sprintf("%s: %s", parts[0], fmt.Sprintf("%v", parts[1:])))
}

func NewError(code int, message string, err error) *APIError {
	return &APIError{
		Code:    code,
		Message: message,
		Err:     err,
		Details: make(map[string]interface{}),
	}
}

func NewErrorWithRequestID(code int, message string, err error, requestID string) *APIError {
	return &APIError{
		Code:      code,
		Message:   message,
		Err:       err,
		RequestID: requestID,
		Details:   make(map[string]interface{}),
	}
}

func NewErrorWithContext(code int, message string, err error, requestID, endpoint, method, resourceType, resourceID string, details map[string]interface{}) *APIError {
	if details == nil {
		details = make(map[string]interface{})
	}
	return &APIError{
		Code:         code,
		Message:      message,
		Err:          err,
		RequestID:    requestID,
		Endpoint:     endpoint,
		Method:       method,
		ResourceType: resourceType,
		ResourceID:   resourceID,
		Details:      details,
	}
}

var (
	ErrNotFound     = NewError(http.StatusNotFound, "resource not found", nil)
	ErrBadRequest   = NewError(http.StatusBadRequest, "bad request", nil)
	ErrUnauthorized = NewError(http.StatusUnauthorized, "unauthorized", nil)
	ErrInternal     = NewError(http.StatusInternalServerError, "internal server error", nil)
)

// WrapError wraps an error with request ID
func WrapError(err *APIError, requestID string) *APIError {
	if err == nil {
		return nil
	}
	return NewErrorWithRequestID(err.Code, err.Message, err.Err, requestID)
}
