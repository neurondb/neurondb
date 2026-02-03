package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
	"time"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

/* Comprehensive test suite for ModelHandlers */

func TestModelHandlers_ListModels_Comprehensive(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	tests := []struct {
		name           string
		queryParams    map[string]string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "list all models",
			queryParams:    map[string]string{},
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if data, ok := result["data"].([]interface{}); !ok {
					t.Error("Expected 'data' array in response")
				} else {
					t.Logf("Found %d models", len(data))
				}
			},
		},
		{
			name:           "list with pagination",
			queryParams:    map[string]string{"limit": "10", "offset": "0"},
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if pagination, ok := result["pagination"].(map[string]interface{}); ok {
					if limit, ok := pagination["limit"].(float64); !ok || limit != 10 {
						t.Errorf("Expected limit 10, got %v", limit)
					}
				}
			},
		},
		{
			name:           "filter by provider",
			queryParams:    map[string]string{"provider": "openai"},
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
			},
		},
		{
			name:           "filter by model type",
			queryParams:    map[string]string{"model_type": "chat"},
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
			},
		},
		{
			name:           "sort by name",
			queryParams:    map[string]string{"sort_by": "name", "sort_order": "ASC"},
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
			},
		},
		{
			name:           "invalid sort field",
			queryParams:    map[string]string{"sort_by": "invalid_field"},
			expectedStatus: http.StatusBadRequest,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var errorResp map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &errorResp); err != nil {
					t.Fatalf("Failed to parse error response: %v", err)
				}
				if code, ok := errorResp["code"].(string); !ok || code != "BAD_REQUEST" {
					t.Errorf("Expected BAD_REQUEST code, got %v", code)
				}
			},
		},
		{
			name:           "invalid sort order",
			queryParams:    map[string]string{"sort_order": "INVALID"},
			expectedStatus: http.StatusBadRequest,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var errorResp map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &errorResp); err != nil {
					t.Fatalf("Failed to parse error response: %v", err)
				}
			},
		},
		{
			name:           "limit too high",
			queryParams:    map[string]string{"limit": "10000"},
			expectedStatus: http.StatusOK, // Should be clamped to max
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if pagination, ok := result["pagination"].(map[string]interface{}); ok {
					if limit, ok := pagination["limit"].(float64); ok && limit > 1000 {
						t.Errorf("Limit should be clamped to 1000, got %v", limit)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := "/api/v1/profiles/" + profile.ID + "/llm-models"
			if len(tt.queryParams) > 0 {
				url += "?"
				first := true
				for k, v := range tt.queryParams {
					if !first {
						url += "&"
					}
					url += k + "=" + v
					first = false
				}
			}

			resp, err := client.Get(url)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)
			if tt.validateFunc != nil {
				tt.validateFunc(t, resp)
			}
		})
	}
}

func TestModelHandlers_AddModel_Comprehensive(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	tests := []struct {
		name           string
		request        map[string]interface{}
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name: "valid OpenAI model",
			request: map[string]interface{}{
				"name":        "gpt-4-test",
				"provider":    "openai",
				"model_type":  "chat",
				"description": "Test GPT-4 model",
				"config": map[string]interface{}{
					"temperature": 0.7,
					"max_tokens":  2000,
				},
			},
			expectedStatus: http.StatusCreated,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var model Model
				if err := testutil.ParseResponse(t, resp, &model); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if model.Name != "gpt-4-test" {
					t.Errorf("Expected name 'gpt-4-test', got %s", model.Name)
				}
				if model.Provider != "openai" {
					t.Errorf("Expected provider 'openai', got %s", model.Provider)
				}
			},
		},
		{
			name: "valid Anthropic model",
			request: map[string]interface{}{
				"name":       "claude-3-test",
				"provider":   "anthropic",
				"model_type": "chat",
			},
			expectedStatus: http.StatusCreated,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var model Model
				if err := testutil.ParseResponse(t, resp, &model); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if model.Provider != "anthropic" {
					t.Errorf("Expected provider 'anthropic', got %s", model.Provider)
				}
			},
		},
		{
			name: "valid embedding model",
			request: map[string]interface{}{
				"name":       "text-embedding-3-small",
				"provider":   "openai",
				"model_type": "embedding",
			},
			expectedStatus: http.StatusCreated,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var model Model
				if err := testutil.ParseResponse(t, resp, &model); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if model.ModelType != "embedding" {
					t.Errorf("Expected model_type 'embedding', got %s", model.ModelType)
				}
			},
		},
		{
			name: "missing name",
			request: map[string]interface{}{
				"provider":   "openai",
				"model_type": "chat",
			},
			expectedStatus: http.StatusBadRequest,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var errorResp map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &errorResp); err != nil {
					t.Fatalf("Failed to parse error response: %v", err)
				}
				if code, ok := errorResp["code"].(string); !ok || code != "VALIDATION_ERROR" {
					t.Errorf("Expected VALIDATION_ERROR code, got %v", code)
				}
			},
		},
		{
			name: "missing provider",
			request: map[string]interface{}{
				"name":       "test-model",
				"model_type": "chat",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "missing model_type",
			request: map[string]interface{}{
				"name":     "test-model",
				"provider": "openai",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid provider",
			request: map[string]interface{}{
				"name":       "test-model",
				"provider":   "invalid-provider",
				"model_type": "chat",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid model_type",
			request: map[string]interface{}{
				"name":       "test-model",
				"provider":   "openai",
				"model_type": "invalid-type",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid model name format",
			request: map[string]interface{}{
				"name":       "test model with spaces!",
				"provider":   "openai",
				"model_type": "chat",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "name too long",
			request: map[string]interface{}{
				"name":       string(make([]byte, 101)),
				"provider":   "openai",
				"model_type": "chat",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid temperature",
			request: map[string]interface{}{
				"name":       "test-model",
				"provider":   "openai",
				"model_type": "chat",
				"config": map[string]interface{}{
					"temperature": 3.0, // Invalid: > 2
				},
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid max_tokens",
			request: map[string]interface{}{
				"name":       "test-model",
				"provider":   "openai",
				"model_type": "chat",
				"config": map[string]interface{}{
					"max_tokens": 200000, // Invalid: > 100000
				},
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid top_p",
			request: map[string]interface{}{
				"name":       "test-model",
				"provider":   "openai",
				"model_type": "chat",
				"config": map[string]interface{}{
					"top_p": 2.0, // Invalid: > 1
				},
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "duplicate name",
			request: map[string]interface{}{
				"name":       "duplicate-test",
				"provider":   "openai",
				"model_type": "chat",
			},
			expectedStatus: http.StatusCreated, // First creation
		},
		{
			name: "duplicate name conflict",
			request: map[string]interface{}{
				"name":       "duplicate-test",
				"provider":   "openai",
				"model_type": "chat",
			},
			expectedStatus: http.StatusConflict, // Second creation should conflict
		},
		{
			name: "description too long",
			request: map[string]interface{}{
				"name":        "test-model",
				"provider":    "openai",
				"model_type":  "chat",
				"description": string(make([]byte, 501)),
			},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/llm-models", tt.request)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)
			if tt.validateFunc != nil {
				tt.validateFunc(t, resp)
			}
		})
	}
}

func TestModelHandlers_SetModelKey_Comprehensive(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Create a test model first
	modelReq := map[string]interface{}{
		"name":       "test-model-key",
		"provider":   "openai",
		"model_type": "chat",
	}
	resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/llm-models", modelReq)
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}
	resp.Body.Close()

	var model Model
	resp, err = client.Get("/api/v1/profiles/" + profile.ID + "/llm-models")
	if err != nil {
		t.Fatalf("Failed to get models: %v", err)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := testutil.ParseResponse(t, resp, &result); err != nil {
		t.Fatalf("Failed to parse models: %v", err)
	}

	models, ok := result["data"].([]interface{})
	if !ok || len(models) == 0 {
		t.Fatal("No models found")
	}

	modelJSON, _ := json.Marshal(models[0])
	json.Unmarshal(modelJSON, &model)

	tests := []struct {
		name           string
		modelName      string
		apiKey         string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "valid API key",
			modelName:      model.Name,
			apiKey:         "sk-test-key-12345678901234567890",
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if message, ok := result["message"].(string); !ok || message != "API key set successfully" {
					t.Errorf("Expected success message, got %v", result)
				}
			},
		},
		{
			name:           "empty API key",
			modelName:      model.Name,
			apiKey:         "",
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:           "API key too short",
			modelName:      model.Name,
			apiKey:         "sk-short",
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:           "model not found",
			modelName:      "non-existent-model",
			apiKey:         "sk-test-key-12345678901234567890",
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := map[string]interface{}{
				"api_key": tt.apiKey,
			}
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/llm-models/"+tt.modelName+"/key", req)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)
			if tt.validateFunc != nil {
				tt.validateFunc(t, resp)
			}
		})
	}
}

func TestModelHandlers_DeleteModel_Comprehensive(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Create a test model
	modelReq := map[string]interface{}{
		"name":       "test-model-delete",
		"provider":   "openai",
		"model_type": "chat",
	}
	resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/llm-models", modelReq)
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}
	defer resp.Body.Close()

	var model Model
	if err := testutil.ParseResponse(t, resp, &model); err != nil {
		t.Fatalf("Failed to parse model: %v", err)
	}

	tests := []struct {
		name           string
		modelID        string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "successful delete",
			modelID:        model.ID,
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				// Verify model is soft-deleted (enabled = false)
				getResp, err := client.Get("/api/v1/profiles/" + profile.ID + "/llm-models/" + model.ID)
				if err == nil {
					defer getResp.Body.Close()
					if getResp.StatusCode == http.StatusOK {
						t.Error("Model should be deleted (not found)")
					}
				}
			},
		},
		{
			name:           "model not found",
			modelID:        "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "invalid UUID",
			modelID:        "invalid-uuid",
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Delete("/api/v1/profiles/" + profile.ID + "/llm-models/" + tt.modelID)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)
			if tt.validateFunc != nil {
				tt.validateFunc(t, resp)
			}
		})
	}
}

func TestModelHandlers_GetModelInfo_Comprehensive(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Create a test model
	modelReq := map[string]interface{}{
		"name":       "test-model-info",
		"provider":   "openai",
		"model_type": "chat",
		"config": map[string]interface{}{
			"temperature": 0.7,
		},
	}
	resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/llm-models", modelReq)
	if err != nil {
		t.Fatalf("Failed to create test model: %v", err)
	}
	defer resp.Body.Close()

	var model Model
	if err := testutil.ParseResponse(t, resp, &model); err != nil {
		t.Fatalf("Failed to parse model: %v", err)
	}

	tests := []struct {
		name           string
		modelID        string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "get existing model",
			modelID:        model.ID,
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result Model
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if result.ID != model.ID {
					t.Errorf("Expected model ID %s, got %s", model.ID, result.ID)
				}
				if result.Name != "test-model-info" {
					t.Errorf("Expected name 'test-model-info', got %s", result.Name)
				}
				if result.Provider != "openai" {
					t.Errorf("Expected provider 'openai', got %s", result.Provider)
				}
			},
		},
		{
			name:           "model not found",
			modelID:        "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "invalid UUID",
			modelID:        "invalid-uuid",
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Get("/api/v1/profiles/" + profile.ID + "/llm-models/" + tt.modelID)
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			testutil.AssertStatus(t, resp, tt.expectedStatus)
			if tt.validateFunc != nil {
				tt.validateFunc(t, resp)
			}
		})
	}
}

/* Performance tests */
func BenchmarkModelHandlers_ListModels(b *testing.B) {
	tdb := testutil.SetupTestDB(&testing.T{})
	defer tdb.CleanupTestDB(&testing.T{})

	client := testutil.NewTestClient(&testing.T{}, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	client.Authenticate(ctx, "testuser", "password123")
	profile, _ := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, _ := client.Get("/api/v1/profiles/" + profile.ID + "/llm-models")
		resp.Body.Close()
	}
}

/* Concurrent access tests */
func TestModelHandlers_ConcurrentAccess(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	client := testutil.NewTestClient(t, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	err := client.Authenticate(ctx, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to authenticate: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Create multiple models concurrently
	concurrency := 10
	errors := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		go func(id int) {
			modelReq := map[string]interface{}{
				"name":       "concurrent-model-" + string(rune(id)),
				"provider":   "openai",
				"model_type": "chat",
			}
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/llm-models", modelReq)
			if err != nil {
				errors <- err
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusCreated {
				errors <- fmt.Errorf("Expected 201, got %d", resp.StatusCode)
			}
		}(i)
	}

	// Wait for all goroutines
	timeout := time.After(5 * time.Second)
	for i := 0; i < concurrency; i++ {
		select {
		case err := <-errors:
			if err != nil {
				t.Errorf("Concurrent operation failed: %v", err)
			}
		case <-timeout:
			t.Fatal("Concurrent operations timed out")
		}
	}
}
