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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, err := ParseRequest(tt.data)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseRequest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && req == nil {
				t.Error("ParseRequest() returned nil request without error")
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRequest(tt.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsNotification(tt.req); got != tt.want {
				t.Errorf("IsNotification() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCreateResponse(t *testing.T) {
	id := json.RawMessage("1")
	result := map[string]string{"test": "value"}

	resp := CreateResponse(id, result)

	if resp.JSONRPC != "2.0" {
		t.Errorf("CreateResponse() JSONRPC = %v, want 2.0", resp.JSONRPC)
	}
	if string(resp.ID) != string(id) {
		t.Errorf("CreateResponse() ID = %v, want %v", resp.ID, id)
	}
	if resp.Error != nil {
		t.Error("CreateResponse() should not have error")
	}
}

func TestCreateErrorResponse(t *testing.T) {
	id := json.RawMessage("1")
	code := ErrCodeInvalidRequest
	message := "test error"

	resp := CreateErrorResponse(id, code, message, nil)

	if resp.JSONRPC != "2.0" {
		t.Errorf("CreateErrorResponse() JSONRPC = %v, want 2.0", resp.JSONRPC)
	}
	if string(resp.ID) != string(id) {
		t.Errorf("CreateErrorResponse() ID = %v, want %v", resp.ID, id)
	}
	if resp.Error == nil {
		t.Error("CreateErrorResponse() should have error")
	}
	if resp.Error.Code != code {
		t.Errorf("CreateErrorResponse() error code = %v, want %v", resp.Error.Code, code)
	}
	if resp.Error.Message != message {
		t.Errorf("CreateErrorResponse() error message = %v, want %v", resp.Error.Message, message)
	}
}


