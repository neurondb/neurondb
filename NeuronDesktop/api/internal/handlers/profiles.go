package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/gorilla/mux"
	"github.com/neurondb/NeuronDesktop/api/internal/auth"
	"github.com/neurondb/NeuronDesktop/api/internal/db"
	"github.com/neurondb/NeuronDesktop/api/internal/utils"
	"github.com/neurondb/NeuronDesktop/api/internal/validation"
	"golang.org/x/crypto/bcrypt"
)

/* ProfileHandlers handles profile-related endpoints */
type ProfileHandlers struct {
	queries          *db.Queries
	neurondbHandlers *NeuronDBHandlers
	agentHandlers    *AgentHandlers
	mcpManager       *MCPManager
}

func isAdmin(ctx context.Context) bool {
	return auth.GetIsAdminFromContext(ctx)
}

/* NewProfileHandlers creates new profile handlers */
func NewProfileHandlers(queries *db.Queries) *ProfileHandlers {
	return &ProfileHandlers{queries: queries}
}

/* SetHandlers sets references to other handlers for client invalidation */
func (h *ProfileHandlers) SetHandlers(neurondbHandlers *NeuronDBHandlers, agentHandlers *AgentHandlers, mcpManager *MCPManager) {
	h.neurondbHandlers = neurondbHandlers
	h.agentHandlers = agentHandlers
	h.mcpManager = mcpManager
}

/* ListProfiles lists profiles for the current user */
func (h *ProfileHandlers) ListProfiles(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	var (
		profiles []db.Profile
		err      error
	)
	if isAdmin(r.Context()) {
		profiles, err = h.queries.ListAllProfiles(r.Context())
	} else {
		profiles, err = h.queries.ListProfiles(r.Context(), userID)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	/* If no profiles exist, create a default profile */
	if len(profiles) == 0 && !isAdmin(r.Context()) {
		mcpConfig := utils.GetDefaultMCPConfig()

		neurondbDSN := utils.GetDefaultNeuronDBDSN()

		agentEndpoint := os.Getenv("NEURONAGENT_ENDPOINT")
		agentAPIKey := os.Getenv("NEURONAGENT_API_KEY")

		defaultProfile := &db.Profile{
			UserID:        userID,
			Name:          "Default",
			NeuronDBDSN:   neurondbDSN,
			MCPConfig:     mcpConfig,
			AgentEndpoint: agentEndpoint,
			AgentAPIKey:   agentAPIKey,
			IsDefault:     true,
		}

		/* Initialize database schema for the default profile's database */
		if err := utils.InitSchema(r.Context(), neurondbDSN); err != nil {
			/* Log error but don't fail - schema might already be initialized */
			fmt.Fprintf(os.Stderr, "Warning: Failed to initialize database schema for default profile (schema may already exist): %v\n", err)
		}

		if err := h.queries.CreateProfile(r.Context(), defaultProfile); err != nil {
			/* Log error but don't fail the request */
			fmt.Fprintf(os.Stderr, "Error: Failed to create default profile during initialization: %v\n", err)
		} else {
			if err := h.queries.SetDefaultProfile(r.Context(), defaultProfile.ID); err != nil {
				fmt.Fprintf(os.Stderr, "Error: Failed to set default profile after creation: %v\n", err)
			}
			/* Automatically create default model configurations */
			if err := utils.CreateDefaultModelsForProfile(r.Context(), h.queries, defaultProfile.ID); err != nil {
				/* Log error but don't fail - user can add models manually */
				fmt.Fprintf(os.Stderr, "Warning: Failed to create default model configurations for profile %s (models can be added manually): %v\n", defaultProfile.ID, err)
			}
			profiles, _ = h.queries.ListProfiles(r.Context(), userID)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(profiles)
}

/* GetProfile gets a single profile */
func (h *ProfileHandlers) GetProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	if err := validation.ValidateUUIDRequired(profileID, "profile_id"); err != nil {
		http.Error(w, fmt.Sprintf("Invalid profile ID: %v", err), http.StatusBadRequest)
		return
	}

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	/* Non-admin users can only access their own profiles */
	if !isAdmin(r.Context()) && profile.UserID != userID {
		http.Error(w, "Access denied", http.StatusForbidden)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(profile)
}

/* CreateProfile creates a new profile
 * NOTE: Profiles are now automatically created during user signup.
 * This endpoint is disabled to prevent manual profile creation. */
func (h *ProfileHandlers) CreateProfile(w http.ResponseWriter, r *http.Request) {
	WriteError(w, r, http.StatusForbidden, fmt.Errorf("profile creation is not allowed. Profiles are automatically created during user signup"), nil)
	return
}

/* UpdateProfile updates a profile */
func (h *ProfileHandlers) UpdateProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	if err := validation.ValidateUUIDRequired(profileID, "profile_id"); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid profile ID: %w", err), nil)
		return
	}

	admin := isAdmin(r.Context())

	var req struct {
		db.Profile
		ProfilePassword string `json:"profile_password"` // Plain password from request
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	existingProfile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	if !admin && existingProfile.UserID != userID {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), nil)
		return
	}

	profile := req.Profile
	profile.ID = profileID
	if admin {
		/* Don't allow admin updates to reassign ownership accidentally */
		profile.UserID = existingProfile.UserID
	} else {
		profile.UserID = userID
	}

	/* Handle profile password update */
	if req.ProfilePassword != "" {
		if profile.ProfileUsername == "" {
			WriteError(w, r, http.StatusBadRequest, fmt.Errorf("profile_username is required when profile_password is set"), nil)
			return
		}
		passwordHash, err := bcrypt.GenerateFromPassword([]byte(req.ProfilePassword), bcrypt.DefaultCost)
		if err != nil {
			WriteError(w, r, http.StatusInternalServerError, fmt.Errorf("failed to hash password"), nil)
			return
		}
		profile.ProfilePassword = string(passwordHash)
	} else if profile.ProfileUsername != "" && profile.ProfileUsername != existingProfile.ProfileUsername {
		/* If username changed but no password provided, require password */
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("profile_password is required when profile_username is set"), nil)
		return
	} else if profile.ProfileUsername == "" {
		/* If clearing username, also clear password */
		profile.ProfilePassword = ""
	}

	if profile.NeuronDBDSN != "" {
		if err := validation.ValidateDSNRequired(profile.NeuronDBDSN, "neurondb_dsn"); err != nil {
			WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid DSN: %w", err), nil)
			return
		}
	}

	if errors := utils.ValidateProfile(profile.Name, profile.NeuronDBDSN, profile.MCPConfig); len(errors) > 0 {
		WriteValidationErrors(w, r, errors)
		return
	}

	/* Check if connection details changed - if so, invalidate clients */
	dsnChanged := existingProfile.NeuronDBDSN != profile.NeuronDBDSN
	agentEndpointChanged := existingProfile.AgentEndpoint != profile.AgentEndpoint
	mcpConfigChanged := !mapsEqual(existingProfile.MCPConfig, profile.MCPConfig)

	if err := h.queries.UpdateProfile(r.Context(), &profile); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	/* Invalidate clients if connection details changed */
	if dsnChanged && h.neurondbHandlers != nil {
		h.neurondbHandlers.InvalidateClient(profileID)
	}
	if agentEndpointChanged && h.agentHandlers != nil {
		h.agentHandlers.InvalidateClient(profileID)
	}
	if mcpConfigChanged && h.mcpManager != nil {
		h.mcpManager.InvalidateClient(profileID)
	}

	WriteSuccess(w, profile, http.StatusOK)
}

/* DeleteProfile deletes a profile */
func (h *ProfileHandlers) DeleteProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		http.Error(w, "Unauthorized", http.StatusUnauthorized)
		return
	}

	if err := validation.ValidateUUIDRequired(profileID, "profile_id"); err != nil {
		http.Error(w, fmt.Sprintf("Invalid profile ID: %v", err), http.StatusBadRequest)
		return
	}

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		http.Error(w, "Profile not found", http.StatusNotFound)
		return
	}

	/* Non-admin users can only delete their own profiles */
	if !isAdmin(r.Context()) && profile.UserID != userID {
		http.Error(w, "Access denied", http.StatusForbidden)
		return
	}

	if err := h.queries.DeleteProfile(r.Context(), profileID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	/* Invalidate all clients for this profile */
	if h.neurondbHandlers != nil {
		h.neurondbHandlers.InvalidateClient(profileID)
	}
	if h.agentHandlers != nil {
		h.agentHandlers.InvalidateClient(profileID)
	}
	if h.mcpManager != nil {
		h.mcpManager.InvalidateClient(profileID)
	}

	w.WriteHeader(http.StatusNoContent)
}

/* ExportProfile exports a profile as JSON */
func (h *ProfileHandlers) ExportProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	/* Non-admin users can only export their own profiles */
	if !isAdmin(r.Context()) && profile.UserID != userID {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), nil)
		return
	}

	/* Create export data (exclude sensitive fields or use placeholders) */
	exportData := map[string]interface{}{
		"version":    "1.0",
		"type":       "profile",
		"exported_at": os.Getenv("TZ"),
		"profile": map[string]interface{}{
			"name":              profile.Name,
			"neurondb_dsn":      profile.NeuronDBDSN,
			"mcp_config":        profile.MCPConfig,
			"agent_endpoint":    profile.AgentEndpoint,
			"default_collection": profile.DefaultCollection,
			/* Note: We don't export passwords or API keys for security */
			"profile_username": profile.ProfileUsername,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=profile-%s-%s.json", profile.Name, profileID[:8]))
	json.NewEncoder(w).Encode(exportData)
}

/* CloneProfile clones an existing profile */
func (h *ProfileHandlers) CloneProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	/* Get original profile */
	originalProfile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	/* Check ownership or admin */
	if !isAdmin(r.Context()) && originalProfile.UserID != userID {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), nil)
		return
	}

	/* Create new profile with same settings */
	newProfile := &db.Profile{
		UserID:          userID,
		Name:            originalProfile.Name + " (Copy)",
		NeuronDBDSN:     originalProfile.NeuronDBDSN,
		MCPConfig:       originalProfile.MCPConfig,
		AgentEndpoint:   originalProfile.AgentEndpoint,
		AgentAPIKey:     originalProfile.AgentAPIKey,
		DefaultCollection: originalProfile.DefaultCollection,
		IsDefault:       false,
	}

	if err := h.queries.CreateProfile(r.Context(), newProfile); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, newProfile, http.StatusCreated)
}

/* ValidateProfile validates a profile configuration */
func (h *ProfileHandlers) ValidateProfile(w http.ResponseWriter, r *http.Request) {
	var req struct {
		NeuronDBDSN   string                 `json:"neurondb_dsn"`
		MCPConfig     map[string]interface{} `json:"mcp_config"`
		AgentEndpoint string                 `json:"agent_endpoint"`
		AgentAPIKey   string                 `json:"agent_api_key"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid request body"), nil)
		return
	}

	validationErrors := []string{}

	/* Validate NeuronDB DSN */
	if req.NeuronDBDSN == "" {
		validationErrors = append(validationErrors, "NeuronDB DSN is required")
	} else {
		dsnResult := validation.ValidateDSN(req.NeuronDBDSN)
		if !dsnResult.Valid {
			validationErrors = append(validationErrors, fmt.Sprintf("Invalid NeuronDB DSN: %s", dsnResult.Error))
		}
	}

	/* Validate MCP config if provided */
	if req.MCPConfig != nil {
		if command, ok := req.MCPConfig["command"].(string); ok && command == "" {
			validationErrors = append(validationErrors, "MCP command cannot be empty if MCP config is provided")
		}
	}

	/* Validate Agent endpoint if provided */
	if req.AgentEndpoint != "" {
		if !validation.ValidateURL(req.AgentEndpoint) {
			validationErrors = append(validationErrors, "Invalid agent endpoint URL")
		}
	}

	if len(validationErrors) > 0 {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("validation failed"), map[string]interface{}{
			"errors": validationErrors,
		})
		return
	}

	WriteSuccess(w, map[string]interface{}{
		"valid":   true,
		"message": "Profile configuration is valid",
	}, http.StatusOK)
}

/* HealthCheckProfile checks the health of a profile's connections */
func (h *ProfileHandlers) HealthCheckProfile(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	profileID := vars["id"]

	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	/* Get profile */
	profile, err := h.queries.GetProfile(r.Context(), profileID)
	if err != nil {
		WriteError(w, r, http.StatusNotFound, fmt.Errorf("profile not found"), nil)
		return
	}

	/* Check ownership or admin */
	if !isAdmin(r.Context()) && profile.UserID != userID {
		WriteError(w, r, http.StatusForbidden, fmt.Errorf("access denied"), nil)
		return
	}

	health := map[string]interface{}{
		"profile_id": profileID,
		"checks":     map[string]interface{}{},
	}

	/* Check NeuronDB connection */
	neurondbHealth := map[string]interface{}{
		"status": "unknown",
	}
	if profile.NeuronDBDSN != "" {
		dsnResult := validation.ValidateDSN(profile.NeuronDBDSN)
		if dsnResult.Valid {
			neurondbHealth["status"] = "valid_dsn"
			neurondbHealth["message"] = "DSN format is valid"
		} else {
			neurondbHealth["status"] = "invalid_dsn"
			neurondbHealth["error"] = dsnResult.Error
		}
	} else {
		neurondbHealth["status"] = "not_configured"
	}
	health["checks"].(map[string]interface{})["neurondb"] = neurondbHealth

	/* Check MCP connection */
	mcpHealth := map[string]interface{}{
		"status": "not_configured",
	}
	if profile.MCPConfig != nil {
		if command, ok := profile.MCPConfig["command"].(string); ok && command != "" {
			mcpHealth["status"] = "configured"
			mcpHealth["command"] = command
		}
	}
	health["checks"].(map[string]interface{})["mcp"] = mcpHealth

	/* Check Agent connection */
	agentHealth := map[string]interface{}{
		"status": "not_configured",
	}
	if profile.AgentEndpoint != "" {
		if validation.ValidateURL(profile.AgentEndpoint) {
			agentHealth["status"] = "valid_endpoint"
			agentHealth["endpoint"] = profile.AgentEndpoint
		} else {
			agentHealth["status"] = "invalid_endpoint"
		}
	}
	health["checks"].(map[string]interface{})["agent"] = agentHealth

	WriteSuccess(w, health, http.StatusOK)
}

/* ImportProfile imports a profile from JSON */
func (h *ProfileHandlers) ImportProfile(w http.ResponseWriter, r *http.Request) {
	userID, ok := auth.GetUserIDFromContext(r.Context())
	if !ok {
		WriteError(w, r, http.StatusUnauthorized, fmt.Errorf("unauthorized"), nil)
		return
	}

	var importData struct {
		Version string                 `json:"version"`
		Type    string                 `json:"type"`
		Profile map[string]interface{} `json:"profile"`
	}

	if err := json.NewDecoder(r.Body).Decode(&importData); err != nil {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err), nil)
		return
	}

	if importData.Type != "profile" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid import type: expected 'profile'"), nil)
		return
	}

	profileData := importData.Profile
	name, _ := profileData["name"].(string)
	if name == "" {
		WriteError(w, r, http.StatusBadRequest, fmt.Errorf("profile name is required"), nil)
		return
	}

	existingProfiles, err := h.queries.ListProfiles(r.Context(), userID)
	if err == nil {
		for _, p := range existingProfiles {
			if p.Name == name {
				WriteError(w, r, http.StatusConflict, fmt.Errorf("profile with name '%s' already exists", name), nil)
				return
			}
		}
	}

	newProfile := db.Profile{
		UserID:          userID,
		Name:            name,
		NeuronDBDSN:     getString(profileData, "neurondb_dsn"),
		MCPConfig:       getMap(profileData, "mcp_config"),
		AgentEndpoint:   getString(profileData, "agent_endpoint"),
		DefaultCollection: getString(profileData, "default_collection"),
		ProfileUsername: getString(profileData, "profile_username"),
		IsDefault:       false, /* Don't import as default */
	}

	if errors := utils.ValidateProfile(newProfile.Name, newProfile.NeuronDBDSN, newProfile.MCPConfig); len(errors) > 0 {
		WriteValidationErrors(w, r, errors)
		return
	}

	if newProfile.NeuronDBDSN != "" {
		if err := validation.ValidateDSNRequired(newProfile.NeuronDBDSN, "neurondb_dsn"); err != nil {
			WriteError(w, r, http.StatusBadRequest, fmt.Errorf("invalid DSN: %w", err), nil)
			return
		}
	}

	if err := h.queries.CreateProfile(r.Context(), &newProfile); err != nil {
		WriteError(w, r, http.StatusInternalServerError, err, nil)
		return
	}

	WriteSuccess(w, newProfile, http.StatusCreated)
}

/* Helper functions for import */
func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getMap(m map[string]interface{}, key string) map[string]interface{} {
	if v, ok := m[key].(map[string]interface{}); ok {
		return v
	}
	return nil
}

/* mapsEqual compares two maps for equality (used for MCP config comparison) */
func mapsEqual(a, b map[string]interface{}) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if bv, ok := b[k]; !ok || !interfaceEqual(v, bv) {
			return false
		}
	}
	return true
}

/* interfaceEqual compares two interface{} values for equality */
func interfaceEqual(a, b interface{}) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	/* Simple comparison - for complex types this may need enhancement */
	return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
}