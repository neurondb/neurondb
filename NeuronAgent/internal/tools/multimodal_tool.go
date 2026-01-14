/*-------------------------------------------------------------------------
 *
 * multimodal_tool.go
 *    Multi-modal tool for image, code, and audio processing
 *
 * Provides agent access to enhanced multi-modal processing capabilities
 * including image analysis, code execution, and audio transcription.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/tools/multimodal_tool.go
 *
 *-------------------------------------------------------------------------
 */

package tools

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/multimodal"
)

/* MultimodalTool provides multi-modal processing capabilities */
type MultimodalTool struct {
	processor *multimodal.EnhancedMultimodalProcessor
}

/* NewMultimodalTool creates a new multimodal tool */
func NewMultimodalTool() *MultimodalTool {
	return &MultimodalTool{
		processor: multimodal.NewEnhancedMultimodalProcessor(),
	}
}

/* Execute executes a multi-modal operation */
func (t *MultimodalTool) Execute(ctx context.Context, tool *db.Tool, args map[string]interface{}) (string, error) {
	action, ok := args["action"].(string)
	if !ok {
		return "", fmt.Errorf("multimodal tool requires action parameter")
	}

	switch action {
	case "process_image":
		return t.processImage(ctx, args)
	case "process_code":
		return t.processCode(ctx, args)
	case "process_audio":
		return t.processAudio(ctx, args)
	default:
		return "", fmt.Errorf("unknown multimodal action: %s", action)
	}
}

/* processImage processes an image */
func (t *MultimodalTool) processImage(ctx context.Context, args map[string]interface{}) (string, error) {
	imageDataStr, ok := args["image_data"].(string)
	if !ok {
		return "", fmt.Errorf("process_image requires image_data parameter (base64 encoded)")
	}

	/* Decode base64 image data */
	imageData, err := base64.StdEncoding.DecodeString(imageDataStr)
	if err != nil {
		return "", fmt.Errorf("invalid base64 image data: %w", err)
	}

	task := "analyze"
	if taskStr, ok := args["task"].(string); ok {
		task = taskStr
	}

	result, err := t.processor.ProcessImage(ctx, imageData, task)
	if err != nil {
		return "", fmt.Errorf("image processing failed: %w", err)
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* processCode processes code */
func (t *MultimodalTool) processCode(ctx context.Context, args map[string]interface{}) (string, error) {
	code, ok := args["code"].(string)
	if !ok {
		return "", fmt.Errorf("process_code requires code parameter")
	}

	language := "python"
	if lang, ok := args["language"].(string); ok {
		language = lang
	}

	task := "analyze"
	if taskStr, ok := args["task"].(string); ok {
		task = taskStr
	}

	result, err := t.processor.ProcessCode(ctx, code, language, task)
	if err != nil {
		return "", fmt.Errorf("code processing failed: %w", err)
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* processAudio processes audio */
func (t *MultimodalTool) processAudio(ctx context.Context, args map[string]interface{}) (string, error) {
	audioDataStr, ok := args["audio_data"].(string)
	if !ok {
		return "", fmt.Errorf("process_audio requires audio_data parameter (base64 encoded)")
	}

	/* Decode base64 audio data */
	audioData, err := base64.StdEncoding.DecodeString(audioDataStr)
	if err != nil {
		return "", fmt.Errorf("invalid base64 audio data: %w", err)
	}

	task := "transcribe"
	if taskStr, ok := args["task"].(string); ok {
		task = taskStr
	}

	result, err := t.processor.ProcessAudio(ctx, audioData, task)
	if err != nil {
		return "", fmt.Errorf("audio processing failed: %w", err)
	}

	resultJSON, _ := json.Marshal(result)
	return string(resultJSON), nil
}

/* Validate validates tool arguments */
func (t *MultimodalTool) Validate(args map[string]interface{}, schema map[string]interface{}) error {
	action, ok := args["action"].(string)
	if !ok {
		return fmt.Errorf("action parameter required")
	}

	validActions := map[string]bool{
		"process_image": true,
		"process_code":  true,
		"process_audio": true,
	}

	if !validActions[action] {
		return fmt.Errorf("invalid action: %s", action)
	}

	return nil
}
