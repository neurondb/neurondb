package handlers

import (
	"context"
	"net/http"
	"testing"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

/* Comprehensive test suite for ObservabilityHandlers */

func TestObservabilityHandlers_GetDBHealth_Comprehensive(t *testing.T) {
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
		profileID      string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "get database health",
			profileID:      profile.ID,
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var health map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &health); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if status, ok := health["status"].(string); !ok {
					t.Error("Expected status in response")
				} else {
					t.Logf("Database status: %s", status)
				}
				if version, ok := health["version"].(string); ok && version != "" {
					t.Logf("Database version: %s", version)
				}
				if connections, ok := health["connections"].(float64); ok {
					t.Logf("Active connections: %.0f", connections)
				}
			},
		},
		{
			name:           "profile not found",
			profileID:      "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "invalid profile ID",
			profileID:      "invalid-uuid",
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Get("/api/v1/profiles/" + tt.profileID + "/observability/db-health")
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

func TestObservabilityHandlers_GetIndexHealth_Comprehensive(t *testing.T) {
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
		profileID      string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "get index health",
			profileID:      profile.ID,
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var indexes []interface{}
				if err := testutil.ParseResponse(t, resp, &indexes); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				t.Logf("Found %d indexes", len(indexes))
				for i, idx := range indexes {
					if idxMap, ok := idx.(map[string]interface{}); ok {
						t.Logf("Index %d: %+v", i, idxMap)
					}
				}
			},
		},
		{
			name:           "profile not found",
			profileID:      "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Get("/api/v1/profiles/" + tt.profileID + "/observability/indexes")
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

func TestObservabilityHandlers_GetWorkerStatus_Comprehensive(t *testing.T) {
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
		profileID      string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "get worker status",
			profileID:      profile.ID,
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var workers []interface{}
				if err := testutil.ParseResponse(t, resp, &workers); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				t.Logf("Found %d workers", len(workers))
				for i, worker := range workers {
					if workerMap, ok := worker.(map[string]interface{}); ok {
						t.Logf("Worker %d: %+v", i, workerMap)
					}
				}
			},
		},
		{
			name:           "profile not found",
			profileID:      "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Get("/api/v1/profiles/" + tt.profileID + "/observability/workers")
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

func TestObservabilityHandlers_GetUsageStats_Comprehensive(t *testing.T) {
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
		profileID      string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "get usage statistics",
			profileID:      profile.ID,
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var stats map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &stats); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if totalRequests, ok := stats["total_requests"].(float64); ok {
					t.Logf("Total requests: %.0f", totalRequests)
				}
				if errors, ok := stats["errors"].(float64); ok {
					t.Logf("Errors: %.0f", errors)
				}
				if avgDuration, ok := stats["avg_duration_ms"].(float64); ok {
					t.Logf("Average duration: %.2f ms", avgDuration)
				}
				if totalTokens, ok := stats["total_tokens"].(float64); ok {
					t.Logf("Total tokens: %.0f", totalTokens)
				}
			},
		},
		{
			name:           "profile not found",
			profileID:      "00000000-0000-0000-0000-000000000000",
			expectedStatus: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Get("/api/v1/profiles/" + tt.profileID + "/observability/usage")
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
func BenchmarkObservabilityHandlers_GetDBHealth(b *testing.B) {
	tdb := testutil.SetupTestDB(&testing.T{})
	defer tdb.CleanupTestDB(&testing.T{})

	client := testutil.NewTestClient(&testing.T{}, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	client.Authenticate(ctx, "testuser", "password123")
	profile, _ := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, _ := client.Get("/api/v1/profiles/" + profile.ID + "/observability/db-health")
		resp.Body.Close()
	}
}






