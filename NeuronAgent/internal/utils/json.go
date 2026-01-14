/*-------------------------------------------------------------------------
 *
 * json.go
 *    JSON marshaling and unmarshaling utilities for NeuronAgent
 *
 * Provides safe JSON encoding and decoding functions with error handling
 * and type conversion utilities.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/utils/json.go
 *
 *-------------------------------------------------------------------------
 */

package utils

import (
	"encoding/json"
	"fmt"
)

/* MarshalJSON safely marshals an object to JSON */
func MarshalJSON(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

/* UnmarshalJSON safely unmarshals JSON to an object */
func UnmarshalJSON(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}

/* MarshalJSONString marshals to JSON string */
func MarshalJSONString(v interface{}) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

/* UnmarshalJSONString unmarshals from JSON string */
func UnmarshalJSONString(s string, v interface{}) error {
	return json.Unmarshal([]byte(s), v)
}

/* PrettyJSON returns pretty-printed JSON */
func PrettyJSON(v interface{}) ([]byte, error) {
	return json.MarshalIndent(v, "", "  ")
}

/* ValidateJSON validates JSON string */
func ValidateJSON(s string) error {
	var v interface{}
	return json.Unmarshal([]byte(s), &v)
}

/* MergeJSON merges two JSON objects */
func MergeJSON(dst, src map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})

	/* Copy dst */
	for k, v := range dst {
		result[k] = v
	}

	/* Merge src (overwrites dst values) */
	for k, v := range src {
		result[k] = v
	}

	return result
}

/* GetJSONField gets a field from JSON object */
func GetJSONField(data map[string]interface{}, path ...string) (interface{}, error) {
	current := data
	for i, key := range path {
		if i == len(path)-1 {
			if val, ok := current[key]; ok {
				return val, nil
			}
			return nil, fmt.Errorf("field not found: %s", key)
		}

		if next, ok := current[key].(map[string]interface{}); ok {
			current = next
		} else {
			return nil, fmt.Errorf("invalid path at: %s", key)
		}
	}

	return nil, fmt.Errorf("empty path")
}
