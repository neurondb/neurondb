package tools

import (
	"testing"
)

func TestBaseTool_ValidateParams(t *testing.T) {
	tool := NewBaseTool(
		"test_tool",
		"Test tool",
		map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"required_field": map[string]interface{}{
					"type": "string",
				},
				"optional_field": map[string]interface{}{
					"type": "number",
				},
			},
			"required": []interface{}{"required_field"},
		},
	)

	tests := []struct {
		name    string
		params  map[string]interface{}
		wantValid bool
	}{
		{
			name: "valid params",
			params: map[string]interface{}{
				"required_field": "test",
				"optional_field": 42.0,
			},
			wantValid: true,
		},
		{
			name: "missing required field",
			params: map[string]interface{}{
				"optional_field": 42.0,
			},
			wantValid: false,
		},
		{
			name: "wrong type",
			params: map[string]interface{}{
				"required_field": 123, // Should be string
			},
			wantValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid, errors := tool.ValidateParams(tt.params, tool.InputSchema())
			if valid != tt.wantValid {
				t.Errorf("ValidateParams() valid = %v, want %v, errors: %v", valid, tt.wantValid, errors)
			}
		})
	}
}

func TestSuccess(t *testing.T) {
	data := map[string]string{"test": "value"}
	metadata := map[string]interface{}{"count": 1}

	result := Success(data, metadata)

	if !result.Success {
		t.Error("Success() should return success=true")
	}
	if result.Error != nil {
		t.Error("Success() should not have error")
	}
	if result.Data == nil {
		t.Error("Success() should have data")
	}
}

func TestError(t *testing.T) {
	message := "test error"
	code := "TEST_ERROR"
	details := map[string]interface{}{"field": "value"}

	result := Error(message, code, details)

	if result.Success {
		t.Error("Error() should return success=false")
	}
	if result.Error == nil {
		t.Error("Error() should have error")
	}
	if result.Error.Message != message {
		t.Errorf("Error() message = %v, want %v", result.Error.Message, message)
	}
	if result.Error.Code != code {
		t.Errorf("Error() code = %v, want %v", result.Error.Code, code)
	}
}


