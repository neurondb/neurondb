/*-------------------------------------------------------------------------
 *
 * types.go
 *    Types for multi-modal support
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/multimodal/types.go
 *
 *-------------------------------------------------------------------------
 */

package multimodal

import "time"

/* MediaType represents the type of media */
type MediaType string

const (
	MediaTypeImage    MediaType = "image"
	MediaTypeAudio    MediaType = "audio"
	MediaTypeVideo    MediaType = "video"
	MediaTypeDocument MediaType = "document"
)

/* MediaFile represents a media file */
type MediaFile struct {
	ID         string                 `json:"id"`
	Type       MediaType              `json:"type"`
	MimeType   string                 `json:"mime_type"`
	Size       int64                  `json:"size"`
	URL        string                 `json:"url,omitempty"`
	Data       []byte                 `json:"data,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	UploadedAt time.Time              `json:"uploaded_at"`
}

/* ImageAnalysis represents image analysis results */
type ImageAnalysis struct {
	Description string                 `json:"description"`
	Objects     []DetectedObject       `json:"objects,omitempty"`
	Text        []ExtractedText        `json:"text,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

/* DetectedObject represents a detected object in an image */
type DetectedObject struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
	Bounds     Bounds  `json:"bounds,omitempty"`
}

/* ExtractedText represents extracted text from an image */
type ExtractedText struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
	Bounds     Bounds  `json:"bounds,omitempty"`
}

/* Bounds represents bounding box coordinates */
type Bounds struct {
	X      float64 `json:"x"`
	Y      float64 `json:"y"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

/* DocumentAnalysis represents document analysis results */
type DocumentAnalysis struct {
	Text     string                 `json:"text"`
	Pages    int                    `json:"pages,omitempty"`
	Language string                 `json:"language,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

/* AudioAnalysis represents audio analysis results */
type AudioAnalysis struct {
	Transcript string                 `json:"transcript"`
	Duration   float64                `json:"duration,omitempty"`
	Language   string                 `json:"language,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

/* VideoAnalysis represents video analysis results */
type VideoAnalysis struct {
	Description string                 `json:"description"`
	Duration    float64                `json:"duration,omitempty"`
	Frames      []FrameAnalysis        `json:"frames,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

/* FrameAnalysis represents analysis of a video frame */
type FrameAnalysis struct {
	Timestamp float64       `json:"timestamp"`
	Analysis  ImageAnalysis `json:"analysis"`
}
