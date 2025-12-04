package mcp

import (
	"encoding/json"
	"testing"
)

func TestParseRequest(t *testing.T) {
	tests := []struct {
		name    string
		data    []byte
		wantErr bool
	}{
		{
			name:    "valid request",
			data:    []byte(`{"jsonrpc":"2.0","id":1,"method":"test","params":{}}`),
			wantErr: false,
		},
		{
			name:    "invalid jsonrpc version",
			data:    []byte(`{"jsonrpc":"1.0","id":1,"method":"test"}`),
			wantErr: true,
		},
		{
			name:    "missing method",
			data:    []byte(`{"jsonrpc":"2.0","id":1}`),
			wantErr: false, // Method validation happens in ValidateRequest
		},
		{
			name:    "notification (no id)",
			data:    []byte(`{"jsonrpc":"2.0","method":"test"}`),
			wantErr: false,
		},
		{
			name:    "invalid JSON",
			data:    []byte(`{invalid json}`),
			wantErr: true,
		},
		{
			name:    "empty JSON",
			data:    []byte(`{}`),
			wantErr: true, // Missing jsonrpc version
		},
		{
			name:    "nil data",
			data:    nil,
			wantErr: true,
		},
		{
			name:    "empty data",
			data:    []byte(``),
			wantErr: true,
		},
		{
			name:    "malformed JSON - unclosed brace",
			data:    []byte(`{"jsonrpc":"2.0"`),
			wantErr: true,
		},
		{
			name:    "null jsonrpc",
			data:    []byte(`{"jsonrpc":null,"method":"test"}`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := ParseRequest(tt.data)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && req == nil {
				t.Fatal("ParseRequest() returned nil request without error")
			}
			if tt.wantErr && err == nil {
				t.Error("ParseRequest() should have returned error")
			}
		})
	}
}

func TestValidateRequest(t *testing.T) {
	tests := []struct {
		name    string
		req     *JSONRPCRequest
		wantErr bool
	}{
		{
			name: "valid request",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
				Method:  "test",
				ID:      json.RawMessage("1"),
			},
			wantErr: false,
		},
		{
			name: "invalid jsonrpc version",
			req: &JSONRPCRequest{
				JSONRPC: "1.0",
				Method:  "test",
			},
			wantErr: true,
		},
		{
			name: "missing method",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
			},
			wantErr: true,
		},
		{
			name: "notification (no id)",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
				Method:  "test",
			},
			wantErr: false,
		},
		{
			name:    "nil request",
			req:     nil,
			wantErr: true,
		},
		{
			name: "empty method",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
				Method:  "",
			},
			wantErr: true,
		},
		{
			name: "empty jsonrpc version",
			req: &JSONRPCRequest{
				JSONRPC: "",
				Method:  "test",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Should not panic
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("ValidateRequest panicked: %v", r)
					}
				}()
				err := ValidateRequest(tt.req)
				if (err != nil) != tt.wantErr {
					t.Errorf("ValidateRequest() error = %v, wantErr %v", err, tt.wantErr)
				}
			}()
		})
	}
}

func TestIsNotification(t *testing.T) {
	tests := []struct {
		name string
		req  *JSONRPCRequest
		want bool
	}{
		{
			name: "notification - no id",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
				Method:  "test",
			},
			want: true,
		},
		{
			name: "notification - null id",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
				Method:  "test",
				ID:      json.RawMessage("null"),
			},
			want: true,
		},
		{
			name: "request with id",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
				Method:  "test",
				ID:      json.RawMessage("1"),
			},
			want: false,
		},
		{
			name: "nil request",
			req:  nil,
			want: true, // nil is treated as notification
		},
		{
			name: "empty id",
			req: &JSONRPCRequest{
				JSONRPC: "2.0",
				Method:  "test",
				ID:      json.RawMessage(""),
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Should not panic
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("IsNotification panicked: %v", r)
					}
				}()
				if got := IsNotification(tt.req); got != tt.want {
					t.Errorf("IsNotification() = %v, want %v", got, tt.want)
				}
			}()
		})
	}
}

func TestCreateResponse(t *testing.T) {
	id := json.RawMessage("1")
	result := map[string]string{"test": "value"}

	resp := CreateResponse(id, result)
	if resp == nil {
		t.Fatal("CreateResponse() returned nil")
	}

	if resp.JSONRPC != "2.0" {
		t.Errorf("CreateResponse() JSONRPC = %v, want 2.0", resp.JSONRPC)
	}
	if string(resp.ID) != string(id) {
		t.Errorf("CreateResponse() ID = %v, want %v", resp.ID, id)
	}
	if resp.Error != nil {
		t.Error("CreateResponse() should not have error")
	}

	// Test with nil result - should not crash
	resp = CreateResponse(id, nil)
	if resp == nil {
		t.Fatal("CreateResponse() returned nil")
	}
	if resp.Result != nil {
		t.Error("CreateResponse() should allow nil result")
	}

	// Test with empty id
	emptyID := json.RawMessage("")
	resp = CreateResponse(emptyID, result)
	if resp == nil {
		t.Fatal("CreateResponse() returned nil")
	}
}

func TestCreateErrorResponse(t *testing.T) {
	id := json.RawMessage("1")
	code := ErrCodeInvalidRequest
	message := "test error"

	resp := CreateErrorResponse(id, code, message, nil)
	if resp == nil {
		t.Fatal("CreateErrorResponse() returned nil")
	}

	if resp.JSONRPC != "2.0" {
		t.Errorf("CreateErrorResponse() JSONRPC = %v, want 2.0", resp.JSONRPC)
	}
	if string(resp.ID) != string(id) {
		t.Errorf("CreateErrorResponse() ID = %v, want %v", resp.ID, id)
	}
	if resp.Error == nil {
		t.Fatal("CreateErrorResponse() should have error")
	}
	if resp.Error.Code != code {
		t.Errorf("CreateErrorResponse() error code = %v, want %v", resp.Error.Code, code)
	}
	if resp.Error.Message != message {
		t.Errorf("CreateErrorResponse() error message = %v, want %v", resp.Error.Message, message)
	}

	// Test with empty message - should not crash
	resp = CreateErrorResponse(id, code, "", nil)
	if resp == nil {
		t.Fatal("CreateErrorResponse() returned nil")
	}
	if resp.Error == nil {
		t.Fatal("CreateErrorResponse() should have error even with empty message")
	}

	// Test with data
	data := map[string]interface{}{"field": "value"}
	resp = CreateErrorResponse(id, code, message, data)
	if resp == nil {
		t.Fatal("CreateErrorResponse() returned nil")
	}
	if resp.Error == nil {
		t.Fatal("CreateErrorResponse() should have error")
	}
}

func TestSerializeResponse(t *testing.T) {
	resp := CreateResponse(json.RawMessage("1"), map[string]string{"test": "value"})
	if resp == nil {
		t.Fatal("CreateResponse() returned nil")
	}

	data, err := SerializeResponse(resp)
	if err != nil {
		t.Fatalf("SerializeResponse() error = %v", err)
	}
	if len(data) == 0 {
		t.Fatal("SerializeResponse() returned empty data")
	}

	// Should be valid JSON
	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("SerializeResponse() produced invalid JSON: %v", err)
	}

	// Test with nil response - should return error
	_, err = SerializeResponse(nil)
	if err == nil {
		t.Error("SerializeResponse() should return error for nil response")
	}
}


