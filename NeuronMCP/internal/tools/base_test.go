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

	if tool == nil {
		t.Fatal("NewBaseTool() returned nil")
	}

	tests := []struct {
		name      string
		params    map[string]interface{}
		wantValid bool
		wantErrors int
	}{
		{
			name: "valid params",
			params: map[string]interface{}{
				"required_field": "test",
				"optional_field": 42.0,
			},
			wantValid:  true,
			wantErrors: 0,
		},
		{
			name: "missing required field",
			params: map[string]interface{}{
				"optional_field": 42.0,
			},
			wantValid:  false,
			wantErrors: 1,
		},
		{
			name: "wrong type",
			params: map[string]interface{}{
				"required_field": 123, // Should be string
			},
			wantValid:  false,
			wantErrors: 1,
		},
		{
			name:      "nil params",
			params:    nil,
			wantValid: false,
			wantErrors: 1,
		},
		{
			name:      "empty params",
			params:    map[string]interface{}{},
			wantValid: false,
			wantErrors: 1,
		},
		{
			name: "wrong type for optional field",
			params: map[string]interface{}{
				"required_field": "test",
				"optional_field": "not a number",
			},
			wantValid:  false,
			wantErrors: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Should not panic
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("ValidateParams panicked: %v", r)
					}
				}()
				valid, errors := tool.ValidateParams(tt.params, tool.InputSchema())
				if valid != tt.wantValid {
					t.Errorf("ValidateParams() valid = %v, want %v, errors: %v", valid, tt.wantValid, errors)
				}
				if len(errors) < tt.wantErrors {
					t.Errorf("ValidateParams() returned %d errors, want at least %d, errors: %v", len(errors), tt.wantErrors, errors)
				}
			}()
		})
	}
}

func TestBaseTool_ValidateParams_NilSchema(t *testing.T) {
	tool := NewBaseTool("test", "test", nil)
	if tool == nil {
		t.Fatal("NewBaseTool() returned nil")
	}

	// Should not panic with nil schema
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("ValidateParams panicked with nil schema: %v", r)
			}
		}()
		valid, errors := tool.ValidateParams(map[string]interface{}{"test": "value"}, nil)
		// With nil schema, validation might pass or fail, but should not crash
		if valid {
			t.Log("ValidateParams() returned valid=true with nil schema (may be acceptable)")
		}
		_ = errors
	}()
}

func TestBaseTool_ValidateParams_InvalidSchema(t *testing.T) {
	tool := NewBaseTool("test", "test", map[string]interface{}{
		"invalid": "schema",
	})

	// Should not panic with invalid schema
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("ValidateParams panicked with invalid schema: %v", r)
			}
		}()
		valid, errors := tool.ValidateParams(map[string]interface{}{"test": "value"}, tool.InputSchema())
		_ = valid
		_ = errors
	}()
}

func TestBaseTool_NilTool(t *testing.T) {
	// Test that methods don't crash on nil tool
	// Note: In Go, calling methods on nil will panic, but we want to ensure
	// our code doesn't create nil tools. This test verifies that NewBaseTool
	// never returns nil.
	tool := NewBaseTool("test", "test", nil)
	if tool == nil {
		t.Fatal("NewBaseTool() should never return nil")
	}
}

func TestSuccess(t *testing.T) {
	data := map[string]string{"test": "value"}
	metadata := map[string]interface{}{"count": 1}

	result := Success(data, metadata)
	if result == nil {
		t.Fatal("Success() returned nil")
	}

	if !result.Success {
		t.Error("Success() should return success=true")
	}
	if result.Error != nil {
		t.Error("Success() should not have error")
	}
	if result.Data == nil {
		t.Error("Success() should have data")
	}

	// Test with nil data
	result = Success(nil, nil)
	if result == nil {
		t.Fatal("Success() returned nil")
	}
	if !result.Success {
		t.Error("Success() should return success=true even with nil data")
	}

	// Test with nil metadata
	result = Success(data, nil)
	if result == nil {
		t.Fatal("Success() returned nil")
	}
	if !result.Success {
		t.Error("Success() should return success=true with nil metadata")
	}
}

func TestError(t *testing.T) {
	message := "test error"
	code := "TEST_ERROR"
	details := map[string]interface{}{"field": "value"}

	result := Error(message, code, details)
	if result == nil {
		t.Fatal("Error() returned nil")
	}

	if result.Success {
		t.Error("Error() should return success=false")
	}
	if result.Error == nil {
		t.Fatal("Error() should have error")
	}
	if result.Error.Message != message {
		t.Errorf("Error() message = %v, want %v", result.Error.Message, message)
	}
	if result.Error.Code != code {
		t.Errorf("Error() code = %v, want %v", result.Error.Code, code)
	}

	// Test with empty message - should not crash
	result = Error("", "", nil)
	if result == nil {
		t.Fatal("Error() returned nil")
	}
	if result.Success {
		t.Error("Error() should return success=false even with empty message")
	}
	if result.Error == nil {
		t.Fatal("Error() should have error even with empty message")
	}

	// Test with nil details
	result = Error(message, code, nil)
	if result == nil {
		t.Fatal("Error() returned nil")
	}
	if result.Error == nil {
		t.Fatal("Error() should have error")
	}
}

func TestNewBaseTool(t *testing.T) {
	// Test with valid inputs
	tool := NewBaseTool("test", "description", map[string]interface{}{})
	if tool == nil {
		t.Fatal("NewBaseTool() returned nil")
	}
	if tool.Name() != "test" {
		t.Errorf("Name() = %v, want test", tool.Name())
	}
	if tool.Description() != "description" {
		t.Errorf("Description() = %v, want description", tool.Description())
	}

	// Test with empty name - should not crash
	tool = NewBaseTool("", "description", map[string]interface{}{})
	if tool == nil {
		t.Fatal("NewBaseTool() returned nil")
	}

	// Test with nil schema - should not crash
	tool2 := NewBaseTool("test", "description", nil)
	if tool2 == nil {
		t.Fatal("NewBaseTool() returned nil")
	}
	if tool2.InputSchema() == nil {
		t.Error("InputSchema() should return nil for nil schema")
	}
}


