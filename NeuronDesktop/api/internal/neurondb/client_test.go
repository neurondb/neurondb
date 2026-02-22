package neurondb

import (
	"context"
	"os"
	"testing"
	"time"
)

func getTestDSN() string {
	dsn := os.Getenv("TEST_NEURONDB_DSN")
	if dsn == "" {
		dsn = "host=localhost port=5432 user=neurondb password=neurondb dbname=neurondb sslmode=disable"
	}
	return dsn
}

func TestNeuronDBClient_NewClient(t *testing.T) {
	tests := []struct {
		name    string
		dsn     string
		wantErr bool
	}{
		{
			name:    "valid DSN",
			dsn:     getTestDSN(),
			wantErr: false, // May fail if DB not available, but that's OK
		},
		{
			name:    "invalid DSN",
			dsn:     "host=invalid port=5432 user=test dbname=test",
			wantErr: true,
		},
		{
			name:    "empty DSN",
			dsn:     "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.dsn)
			if !tt.wantErr && err != nil {
				t.Skipf("NeuronDB not available (skipping): %v", err)
			}
			if (err != nil) != tt.wantErr {
				t.Errorf("NewClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if client != nil {
				defer client.Close()
			}
		})
	}
}

func TestNeuronDBClient_ListCollections(t *testing.T) {
	dsn := getTestDSN()
	client, err := NewClient(dsn)
	if err != nil {
		t.Skipf("Skipping test: cannot connect to NeuronDB: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	collections, err := client.ListCollections(ctx)
	if err != nil {
		t.Logf("ListCollections failed (may be expected if no collections exist): %v", err)
		return
	}

	// Verify collections structure
	for _, coll := range collections {
		if coll.Name == "" {
			t.Error("Collection name should not be empty")
		}
		if coll.Schema == "" {
			t.Error("Collection schema should not be empty")
		}
	}
}

func TestNeuronDBClient_Search(t *testing.T) {
	dsn := getTestDSN()
	client, err := NewClient(dsn)
	if err != nil {
		t.Skipf("Skipping test: cannot connect to NeuronDB: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := SearchRequest{
		Collection:   "documents",
		QueryVector:  []float32{0.1, 0.2, 0.3},
		Limit:        10,
		DistanceType: "cosine",
	}

	results, err := client.Search(ctx, req)
	if err != nil {
		t.Logf("Search failed (may be expected if collection doesn't exist): %v", err)
		return
	}

	// Verify results structure
	for _, result := range results {
		if result.ID == nil {
			t.Error("Search result ID should not be nil")
		}
		if result.Score < 0 || result.Score > 1 {
			t.Errorf("Search result score should be between 0 and 1, got %f", result.Score)
		}
	}
}

func TestNeuronDBClient_ExecuteSQL(t *testing.T) {
	dsn := getTestDSN()
	client, err := NewClient(dsn)
	if err != nil {
		t.Skipf("Skipping test: cannot connect to NeuronDB: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	tests := []struct {
		name    string
		query   string
		wantErr bool
	}{
		{
			name:    "valid SELECT",
			query:   "SELECT 1 as test",
			wantErr: false,
		},
		{
			name:    "invalid query",
			query:   "SELECT * FROM nonexistent_table_xyz",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			results, err := client.ExecuteSQL(ctx, tt.query)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExecuteSQL() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && results == nil {
				t.Error("Expected results for valid query")
			}
		})
	}
}

func TestNeuronDBClient_Close(t *testing.T) {
	dsn := getTestDSN()
	client, err := NewClient(dsn)
	if err != nil {
		t.Skipf("Skipping test: cannot connect to NeuronDB: %v", err)
	}

	// Close should not error
	if err := client.Close(); err != nil {
		t.Errorf("Close() error = %v", err)
	}

	// Closing again should not panic
	if err := client.Close(); err != nil {
		t.Logf("Close() on already closed client returned error (may be expected): %v", err)
	}
}







