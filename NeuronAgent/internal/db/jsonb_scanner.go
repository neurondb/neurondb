package db

import (
	"database/sql/driver"
	"encoding/json"
)

// JSONBMap is a custom type for scanning JSONB into map[string]interface{}
type JSONBMap map[string]interface{}

// Scan implements the sql.Scanner interface for JSONBMap
func (m *JSONBMap) Scan(value interface{}) error {
	// Always initialize to empty map first
	*m = make(JSONBMap)
	
	if value == nil {
		return nil
	}

	var bytes []byte
	switch v := value.(type) {
	case []byte:
		bytes = v
	case string:
		bytes = []byte(v)
	default:
		// For unknown types, just return empty map (don't error)
		return nil
	}

	if len(bytes) == 0 || string(bytes) == "{}" || string(bytes) == "null" {
		return nil
	}

	var result map[string]interface{}
	if err := json.Unmarshal(bytes, &result); err != nil {
		// If unmarshal fails, return empty map instead of error
		// This handles cases where the JSONB might be malformed
		return nil
	}

	*m = JSONBMap(result)
	return nil
}

// Value implements the driver.Valuer interface for JSONBMap
func (m JSONBMap) Value() (driver.Value, error) {
	if m == nil || len(m) == 0 {
		return "{}", nil
	}
	return json.Marshal(m)
}

// ToMap converts JSONBMap to map[string]interface{}
func (m JSONBMap) ToMap() map[string]interface{} {
	if m == nil {
		return make(map[string]interface{})
	}
	return map[string]interface{}(m)
}

// FromMap creates JSONBMap from map[string]interface{}
func FromMap(m map[string]interface{}) JSONBMap {
	if m == nil {
		return make(JSONBMap)
	}
	return JSONBMap(m)
}

