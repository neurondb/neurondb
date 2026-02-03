package db_test

import (
	"context"
	"testing"
	"time"

	"github.com/neurondb/NeuronDesktop/api/internal/db"
	testutil "github.com/neurondb/NeuronDesktop/api/internal/testing"
)

func TestQueries_CreateUser(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	user := &db.User{
		Username:     "testuser",
		PasswordHash: "hashed_password",
		IsAdmin:      false,
	}

	err := tdb.Queries.CreateUser(ctx, user)
	if err != nil {
		t.Fatalf("CreateUser() error = %v", err)
	}

	if user.ID == "" {
		t.Error("Expected user ID to be set")
	}
	if user.CreatedAt.IsZero() {
		t.Error("Expected CreatedAt to be set")
	}
}

func TestQueries_GetUserByUsername(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	// Get user by username
	found, err := tdb.Queries.GetUserByUsername(ctx, "testuser")
	if err != nil {
		t.Fatalf("GetUserByUsername() error = %v", err)
	}

	if found.ID != user.ID {
		t.Errorf("Expected user ID %s, got %s", user.ID, found.ID)
	}
	if found.Username != user.Username {
		t.Errorf("Expected username %s, got %s", user.Username, found.Username)
	}
}

func TestQueries_CreateProfile(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user first
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	profile := &db.Profile{
		UserID:      user.ID,
		Name:        "Test Profile",
		NeuronDBDSN: "host=localhost port=5432 user=neurondb dbname=neurondb",
		MCPConfig: map[string]interface{}{
			"command": "echo",
			"args":    []string{"test"},
		},
		IsDefault: false,
	}

	err = tdb.Queries.CreateProfile(ctx, profile)
	if err != nil {
		t.Fatalf("CreateProfile() error = %v", err)
	}

	if profile.ID == "" {
		t.Error("Expected profile ID to be set")
	}
	if profile.CreatedAt.IsZero() {
		t.Error("Expected CreatedAt to be set")
	}
}

func TestQueries_GetProfile(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user and profile
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, user.ID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Get profile
	found, err := tdb.Queries.GetProfile(ctx, profile.ID)
	if err != nil {
		t.Fatalf("GetProfile() error = %v", err)
	}

	if found.ID != profile.ID {
		t.Errorf("Expected profile ID %s, got %s", profile.ID, found.ID)
	}
	if found.Name != profile.Name {
		t.Errorf("Expected profile name %s, got %s", profile.Name, found.Name)
	}
}

func TestQueries_ListProfiles(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	// Create multiple profiles
	profile1, err := testutil.CreateTestProfile(ctx, tdb.Queries, user.ID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	profile2 := &db.Profile{
		UserID:      user.ID,
		Name:        "Second Profile",
		NeuronDBDSN: "host=localhost port=5432 user=neurondb dbname=neurondb",
		MCPConfig: map[string]interface{}{
			"command": "echo",
		},
		IsDefault: false,
	}
	if err := tdb.Queries.CreateProfile(ctx, profile2); err != nil {
		t.Fatalf("Failed to create second profile: %v", err)
	}

	// List profiles
	profiles, err := tdb.Queries.ListProfiles(ctx, user.ID)
	if err != nil {
		t.Fatalf("ListProfiles() error = %v", err)
	}

	if len(profiles) < 2 {
		t.Errorf("Expected at least 2 profiles, got %d", len(profiles))
	}

	found1 := false
	found2 := false
	for _, p := range profiles {
		if p.ID == profile1.ID {
			found1 = true
		}
		if p.ID == profile2.ID {
			found2 = true
		}
	}

	if !found1 {
		t.Error("Expected to find profile1 in list")
	}
	if !found2 {
		t.Error("Expected to find profile2 in list")
	}
}

func TestQueries_UpdateProfile(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user and profile
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, user.ID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Update profile
	profile.Name = "Updated Profile Name"
	profile.UpdatedAt = time.Now()

	err = tdb.Queries.UpdateProfile(ctx, profile)
	if err != nil {
		t.Fatalf("UpdateProfile() error = %v", err)
	}

	// Verify update
	updated, err := tdb.Queries.GetProfile(ctx, profile.ID)
	if err != nil {
		t.Fatalf("GetProfile() error = %v", err)
	}

	if updated.Name != "Updated Profile Name" {
		t.Errorf("Expected profile name 'Updated Profile Name', got %s", updated.Name)
	}
}

func TestQueries_DeleteProfile(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user and profile
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, user.ID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Delete profile
	err = tdb.Queries.DeleteProfile(ctx, profile.ID)
	if err != nil {
		t.Fatalf("DeleteProfile() error = %v", err)
	}

	// Verify deletion
	_, err = tdb.Queries.GetProfile(ctx, profile.ID)
	if err == nil {
		t.Error("Expected profile to be deleted")
	}
}

func TestQueries_CreateModelConfig(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user and profile
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, user.ID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	config := &db.ModelConfig{
		ProfileID:     profile.ID,
		ModelProvider: "openai",
		ModelName:     "gpt-4",
		APIKey:        "sk-test-key",
		IsDefault:     false,
		IsFree:        false,
	}

	err = tdb.Queries.CreateModelConfig(ctx, config)
	if err != nil {
		t.Fatalf("CreateModelConfig() error = %v", err)
	}

	if config.ID == "" {
		t.Error("Expected model config ID to be set")
	}
	if config.CreatedAt.IsZero() {
		t.Error("Expected CreatedAt to be set")
	}
}

func TestQueries_ListModelConfigs(t *testing.T) {
	tdb := testutil.SetupTestDB(t)
	defer tdb.CleanupTestDB(t)

	ctx := context.Background()

	// Create a user and profile
	user, err := testutil.CreateTestUser(ctx, tdb.Queries, "testuser", "password123")
	if err != nil {
		t.Fatalf("Failed to create test user: %v", err)
	}

	profile, err := testutil.CreateTestProfile(ctx, tdb.Queries, user.ID)
	if err != nil {
		t.Fatalf("Failed to create test profile: %v", err)
	}

	// Create model configs
	config1 := &db.ModelConfig{
		ProfileID:     profile.ID,
		ModelProvider: "openai",
		ModelName:     "gpt-4",
		APIKey:        "sk-test-key-1",
		IsDefault:     false,
	}
	if err := tdb.Queries.CreateModelConfig(ctx, config1); err != nil {
		t.Fatalf("Failed to create model config: %v", err)
	}

	config2 := &db.ModelConfig{
		ProfileID:     profile.ID,
		ModelProvider: "anthropic",
		ModelName:     "claude-3-opus",
		APIKey:        "sk-test-key-2",
		IsDefault:     false,
	}
	if err := tdb.Queries.CreateModelConfig(ctx, config2); err != nil {
		t.Fatalf("Failed to create model config: %v", err)
	}

	// List model configs
	configs, err := tdb.Queries.ListModelConfigs(ctx, profile.ID, false)
	if err != nil {
		t.Fatalf("ListModelConfigs() error = %v", err)
	}

	if len(configs) < 2 {
		t.Errorf("Expected at least 2 model configs, got %d", len(configs))
	}
}
