package agent

import (
	"github.com/neurondb/NeuronAgent/internal/db"
)

// Profile represents a predefined agent profile
type Profile struct {
	Name         string
	Description  string
	SystemPrompt string
	ModelName    string
	Config       map[string]interface{}
	EnabledTools []string
}

// GetDefaultProfiles returns default agent profiles
func GetDefaultProfiles() []Profile {
	return []Profile{
		{
			Name:         "general-assistant",
			Description:  "General purpose assistant for answering questions and helping with tasks",
			SystemPrompt: "You are a helpful, harmless, and honest assistant. Answer questions accurately and helpfully.",
			ModelName:    "gpt-4",
			Config: map[string]interface{}{
				"temperature": 0.7,
				"max_tokens":  1000,
				"top_p":       0.9,
			},
			EnabledTools: []string{"sql", "http"},
		},
		{
			Name:         "code-assistant",
			Description:  "Specialized assistant for code analysis and programming tasks",
			SystemPrompt: "You are an expert programmer. Help with code analysis, debugging, and writing code.",
			ModelName:    "gpt-4",
			Config: map[string]interface{}{
				"temperature": 0.3,
				"max_tokens":  2000,
				"top_p":       0.95,
			},
			EnabledTools: []string{"code", "sql"},
		},
		{
			Name:         "data-analyst",
			Description:  "Assistant specialized in data analysis and SQL queries",
			SystemPrompt: "You are a data analyst. Help with SQL queries, data analysis, and insights.",
			ModelName:    "gpt-4",
			Config: map[string]interface{}{
				"temperature": 0.2,
				"max_tokens":  1500,
				"top_p":       0.9,
			},
			EnabledTools: []string{"sql"},
		},
		{
			Name:         "research-assistant",
			Description:  "Assistant for research and information gathering",
			SystemPrompt: "You are a research assistant. Help gather information, summarize content, and provide insights.",
			ModelName:    "gpt-4",
			Config: map[string]interface{}{
				"temperature": 0.5,
				"max_tokens":  2000,
				"top_p":       0.95,
			},
			EnabledTools: []string{"http", "sql"},
		},
	}
}

// CreateAgentFromProfile creates an agent from a profile
func CreateAgentFromProfile(profile Profile) *db.Agent {
	return &db.Agent{
		Name:         profile.Name,
		Description:  &profile.Description,
		SystemPrompt: profile.SystemPrompt,
		ModelName:    profile.ModelName,
		Config:       profile.Config,
		EnabledTools: profile.EnabledTools,
	}
}

// FindProfile finds a profile by name
func FindProfile(name string) *Profile {
	profiles := GetDefaultProfiles()
	for _, p := range profiles {
		if p.Name == name {
			return &p
		}
	}
	return nil
}

