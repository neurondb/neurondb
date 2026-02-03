package handlers

import (
	"context"
	"fmt"
	"net/http"
	"testing"
	"time"

	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

/* Comprehensive test suite for DatasetHandlers */

func TestDatasetHandlers_IngestDataset_Comprehensive(t *testing.T) {
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
			name: "ingest from file",
			request: map[string]interface{}{
				"source_type":     "file",
				"source_path":     "/tmp/test-data.csv",
				"format":          "csv",
				"table_name":      "test_table",
				"auto_embed":      true,
				"embedding_model": "text-embedding-3-small",
				"create_index":    true,
			},
			expectedStatus: http.StatusAccepted,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if jobID, ok := result["job_id"].(string); !ok || jobID == "" {
					t.Error("Expected job_id in response")
				}
				if status, ok := result["status"].(string); !ok || status != "queued" {
					t.Errorf("Expected status 'queued', got %v", status)
				}
			},
		},
		{
			name: "ingest from URL",
			request: map[string]interface{}{
				"source_type":  "url",
				"source_path":  "https://example.com/data.json",
				"format":       "json",
				"auto_embed":   false,
				"create_index": false,
			},
			expectedStatus: http.StatusAccepted,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
			},
		},
		{
			name: "ingest from S3",
			request: map[string]interface{}{
				"source_type":  "s3",
				"source_path":  "s3://bucket/path/to/data.parquet",
				"format":       "parquet",
				"auto_embed":   true,
				"create_index": true,
			},
			expectedStatus: http.StatusAccepted,
		},
		{
			name: "ingest from GitHub",
			request: map[string]interface{}{
				"source_type":  "github",
				"source_path":  "owner/repo/path/to/data.json",
				"format":       "json",
				"auto_embed":   true,
				"create_index": true,
			},
			expectedStatus: http.StatusAccepted,
		},
		{
			name: "ingest from HuggingFace",
			request: map[string]interface{}{
				"source_type":  "huggingface",
				"source_path":  "dataset-name",
				"auto_embed":   true,
				"create_index": true,
			},
			expectedStatus: http.StatusAccepted,
		},
		{
			name: "missing source_type",
			request: map[string]interface{}{
				"source_path": "/tmp/test.csv",
			},
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
			name: "missing source_path",
			request: map[string]interface{}{
				"source_type": "file",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid source_type",
			request: map[string]interface{}{
				"source_type": "invalid",
				"source_path": "/tmp/test.csv",
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "invalid format",
			request: map[string]interface{}{
				"source_type": "file",
				"source_path": "/tmp/test.csv",
				"format":      "invalid-format",
			},
			expectedStatus: http.StatusAccepted, // Format validation may be lenient
		},
		{
			name: "auto_embed without embedding_model",
			request: map[string]interface{}{
				"source_type":  "file",
				"source_path":  "/tmp/test.csv",
				"auto_embed":   true,
				"create_index": true,
			},
			expectedStatus: http.StatusAccepted, // May use default model
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/neurondb/ingest", tt.request)
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

func TestDatasetHandlers_GetIngestStatus_Comprehensive(t *testing.T) {
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

	// Start an ingestion job
	ingestReq := map[string]interface{}{
		"source_type": "file",
		"source_path": "/tmp/test.csv",
		"format":      "csv",
	}
	resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/neurondb/ingest", ingestReq)
	if err != nil {
		t.Fatalf("Failed to start ingestion: %v", err)
	}
	defer resp.Body.Close()

	var ingestResult map[string]interface{}
	if err := testutil.ParseResponse(t, resp, &ingestResult); err != nil {
		t.Fatalf("Failed to parse ingestion response: %v", err)
	}

	jobID, ok := ingestResult["job_id"].(string)
	if !ok {
		t.Fatal("No job_id in ingestion response")
	}

	tests := []struct {
		name           string
		jobID          string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "get existing job status",
			jobID:          jobID,
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var result map[string]interface{}
				if err := testutil.ParseResponse(t, resp, &result); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if status, ok := result["status"].(string); !ok {
					t.Error("Expected status in response")
				} else {
					t.Logf("Job status: %s", status)
				}
			},
		},
		{
			name:           "job not found",
			jobID:          "non-existent-job-id",
			expectedStatus: http.StatusNotFound,
		},
		{
			name:           "empty job ID",
			jobID:          "",
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := client.Get("/api/v1/profiles/" + profile.ID + "/neurondb/ingest/" + tt.jobID)
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

func TestDatasetHandlers_ListIngestJobs_Comprehensive(t *testing.T) {
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

	// Start multiple ingestion jobs
	for i := 0; i < 3; i++ {
		ingestReq := map[string]interface{}{
			"source_type": "file",
			"source_path": "/tmp/test-" + string(rune(i)) + ".csv",
			"format":      "csv",
		}
		resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/neurondb/ingest", ingestReq)
		if err == nil {
			resp.Body.Close()
		}
		time.Sleep(100 * time.Millisecond) // Small delay between jobs
	}

	tests := []struct {
		name           string
		queryParams    map[string]string
		expectedStatus int
		validateFunc   func(t *testing.T, resp *http.Response)
	}{
		{
			name:           "list all jobs",
			queryParams:    map[string]string{},
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var jobs []interface{}
				if err := testutil.ParseResponse(t, resp, &jobs); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				t.Logf("Found %d ingestion jobs", len(jobs))
			},
		},
		{
			name:           "list with limit",
			queryParams:    map[string]string{"limit": "2"},
			expectedStatus: http.StatusOK,
			validateFunc: func(t *testing.T, resp *http.Response) {
				var jobs []interface{}
				if err := testutil.ParseResponse(t, resp, &jobs); err != nil {
					t.Fatalf("Failed to parse response: %v", err)
				}
				if len(jobs) > 2 {
					t.Errorf("Expected at most 2 jobs, got %d", len(jobs))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := "/api/v1/profiles/" + profile.ID + "/neurondb/ingest"
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

/* Performance tests */
func BenchmarkDatasetHandlers_IngestDataset(b *testing.B) {
	tdb := testutil.SetupTestDB(&testing.T{})
	defer tdb.CleanupTestDB(&testing.T{})

	client := testutil.NewTestClient(&testing.T{}, tdb.Queries)
	defer client.Server.Close()

	ctx := context.Background()
	client.Authenticate(ctx, "testuser", "password123")
	profile, _ := testutil.CreateTestProfile(ctx, tdb.Queries, client.UserID)

	req := map[string]interface{}{
		"source_type": "file",
		"source_path": "/tmp/test.csv",
		"format":      "csv",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resp, _ := client.Post("/api/v1/profiles/"+profile.ID+"/neurondb/ingest", req)
		resp.Body.Close()
	}
}

/* Concurrent ingestion tests */
func TestDatasetHandlers_ConcurrentIngestion(t *testing.T) {
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

	concurrency := 5
	errors := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		go func(id int) {
			req := map[string]interface{}{
				"source_type": "file",
				"source_path": "/tmp/concurrent-test-" + string(rune(id)) + ".csv",
				"format":      "csv",
			}
			resp, err := client.Post("/api/v1/profiles/"+profile.ID+"/neurondb/ingest", req)
			if err != nil {
				errors <- err
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusAccepted {
				errors <- fmt.Errorf("Expected 202, got %d", resp.StatusCode)
			}
		}(i)
	}

	timeout := time.After(5 * time.Second)
	for i := 0; i < concurrency; i++ {
		select {
		case err := <-errors:
			if err != nil {
				t.Errorf("Concurrent ingestion failed: %v", err)
			}
		case <-timeout:
			t.Fatal("Concurrent ingestion timed out")
		}
	}
}
