/*-------------------------------------------------------------------------
 *
 * processor.go
 *    Multi-modal media processor
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/multimodal/processor.go
 *
 *-------------------------------------------------------------------------
 */

package multimodal

import (
	"context"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

/* MediaProcessor processes different types of media */
type MediaProcessor struct {
	imageProcessor    ImageProcessor
	documentProcessor DocumentProcessor
	audioProcessor    AudioProcessor
	videoProcessor    VideoProcessor
}

/* NewMediaProcessor creates a new media processor */
func NewMediaProcessor() *MediaProcessor {
	return &MediaProcessor{
		imageProcessor:    NewImageProcessor(),
		documentProcessor: NewDocumentProcessor(),
		audioProcessor:    NewAudioProcessor(),
		videoProcessor:    NewVideoProcessor(),
	}
}

/* Process processes a media file */
func (mp *MediaProcessor) Process(ctx context.Context, file *MediaFile) (interface{}, error) {
	switch file.Type {
	case MediaTypeImage:
		return mp.imageProcessor.Process(ctx, file)
	case MediaTypeDocument:
		return mp.documentProcessor.Process(ctx, file)
	case MediaTypeAudio:
		return mp.audioProcessor.Process(ctx, file)
	case MediaTypeVideo:
		return mp.videoProcessor.Process(ctx, file)
	default:
		return nil, fmt.Errorf("unsupported media type: %s", file.Type)
	}
}

/* ImageProcessor processes images */
type ImageProcessor interface {
	Process(ctx context.Context, file *MediaFile) (*ImageAnalysis, error)
}

/* DocumentProcessor processes documents */
type DocumentProcessor interface {
	Process(ctx context.Context, file *MediaFile) (*DocumentAnalysis, error)
}

/* AudioProcessor processes audio */
type AudioProcessor interface {
	Process(ctx context.Context, file *MediaFile) (*AudioAnalysis, error)
}

/* VideoProcessor processes video */
type VideoProcessor interface {
	Process(ctx context.Context, file *MediaFile) (*VideoAnalysis, error)
}

/* NewImageProcessor creates a new image processor */
func NewImageProcessor() ImageProcessor {
	return &basicImageProcessor{}
}

/* NewDocumentProcessor creates a new document processor */
func NewDocumentProcessor() DocumentProcessor {
	return &basicDocumentProcessor{}
}

/* NewAudioProcessor creates a new audio processor */
func NewAudioProcessor() AudioProcessor {
	return &basicAudioProcessor{}
}

/* NewVideoProcessor creates a new video processor */
func NewVideoProcessor() VideoProcessor {
	return &basicVideoProcessor{}
}

/* Basic implementations (can be enhanced with actual ML models) */
type basicImageProcessor struct{}

func (p *basicImageProcessor) Process(ctx context.Context, file *MediaFile) (*ImageAnalysis, error) {
	metadata := map[string]interface{}{
		"size":      file.Size,
		"mime_type": file.MimeType,
	}

	/* Extract image dimensions and format */
	var width, height int
	var format string

	/* Try to read image from URL or data */
	var img image.Image
	var err error

	if file.URL != "" {
		/* Try to open from URL */
		img, format, err = p.loadImageFromURL(ctx, file.URL)
	} else if len(file.Data) > 0 {
		/* Decode from base64 or raw data */
		img, format, err = p.loadImageFromData(file.Data)
	}

	if err == nil && img != nil {
		bounds := img.Bounds()
		width = bounds.Dx()
		height = bounds.Dy()
		metadata["width"] = width
		metadata["height"] = height
		metadata["format"] = format
		metadata["aspect_ratio"] = float64(width) / float64(height)
	}

	/* Try OCR if Tesseract is available */
	ocrText := ""
	if file.URL != "" || len(file.Data) > 0 {
		ocrText, _ = p.extractOCR(ctx, file)
		if ocrText != "" {
			metadata["has_text"] = true
			metadata["ocr_text_length"] = len(ocrText)
		}
	}

	/* Generate description using vision model integration if available */
	description := fmt.Sprintf("Image: %dx%d pixels, format: %s", width, height, format)
	if ocrText != "" {
		description += fmt.Sprintf(", contains text: %s", truncateString(ocrText, 100))
	}

	/* Try to use NeuronDB vision function if database connection is available */
	/* This would be set if processor is initialized with database access */
	/* For now, use basic description but allow metadata override */
	if file.Metadata != nil {
		if preAnalyzed, ok := file.Metadata["description"].(string); ok {
			description = preAnalyzed
		}
		/* Check if vision analysis was already done */
		if visionDesc, ok := file.Metadata["vision_description"].(string); ok {
			description = visionDesc
		}
	}

	/* Build text extraction results */
	var extractedText []ExtractedText
	if ocrText != "" {
		extractedText = []ExtractedText{
			{
				Text:       ocrText,
				Confidence: 0.8, /* Default confidence for Tesseract */
			},
		}
	}

	return &ImageAnalysis{
		Description: description,
		Text:        extractedText,
		Metadata:    metadata,
	}, nil
}

/* loadImageFromURL loads image from URL */
func (p *basicImageProcessor) loadImageFromURL(ctx context.Context, url string) (image.Image, string, error) {
	/* For HTTP URLs, would use http.Get */
	/* For file URLs, use file system */
	if strings.HasPrefix(url, "file://") {
		path := strings.TrimPrefix(url, "file://")
		file, err := os.Open(path)
		if err != nil {
			return nil, "", err
		}
		defer file.Close()
		img, format, err := image.Decode(file)
		return img, format, err
	}
	return nil, "", fmt.Errorf("unsupported URL scheme")
}

/* loadImageFromData loads image from data */
func (p *basicImageProcessor) loadImageFromData(data []byte) (image.Image, string, error) {
	reader := io.NopCloser(strings.NewReader(string(data)))
	img, format, err := image.Decode(reader)
	return img, format, err
}

/* extractOCR extracts text from image using Tesseract */
func (p *basicImageProcessor) extractOCR(ctx context.Context, file *MediaFile) (string, error) {
	/* Check if Tesseract is available */
	if _, err := exec.LookPath("tesseract"); err != nil {
		return "", nil /* Tesseract not available, skip OCR */
	}

	/* Create temporary file for image */
	var tempFile *os.File
	var err error

	if file.URL != "" && strings.HasPrefix(file.URL, "file://") {
		path := strings.TrimPrefix(file.URL, "file://")
		tempFile, err = os.Open(path)
		if err != nil {
			return "", err
		}
		defer tempFile.Close()
	} else {
		/* Write data to temp file */
		tempFile, err = os.CreateTemp("", "ocr-image-*")
		if err != nil {
			return "", err
		}
		defer os.Remove(tempFile.Name())
		defer tempFile.Close()

		if len(file.Data) > 0 {
			/* Try to decode base64 */
			var imageData []byte
			if strings.HasPrefix(string(file.Data), "data:") {
				/* Data URL format */
				parts := strings.Split(string(file.Data), ",")
				if len(parts) == 2 {
					imageData, err = base64.StdEncoding.DecodeString(parts[1])
					if err != nil {
						return "", err
					}
				}
			} else {
				imageData = file.Data
			}

			if _, err := tempFile.Write(imageData); err != nil {
				return "", err
			}
		} else {
			return "", fmt.Errorf("no image data available")
		}
	}

	/* Run Tesseract OCR */
	cmd := exec.CommandContext(ctx, "tesseract", tempFile.Name(), "stdout", "-l", "eng")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("tesseract failed: %w", err)
	}

	return strings.TrimSpace(string(output)), nil
}

/* truncateString truncates a string to max length */
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

type basicDocumentProcessor struct{}

func (p *basicDocumentProcessor) Process(ctx context.Context, file *MediaFile) (*DocumentAnalysis, error) {
	/* Placeholder implementation - in production, integrate with OCR/document parsing */
	return &DocumentAnalysis{
		Text: "Document processing not yet implemented. Integrate with OCR libraries like Tesseract, document parsers, etc.",
		Metadata: map[string]interface{}{
			"size":      file.Size,
			"mime_type": file.MimeType,
		},
	}, nil
}

type basicAudioProcessor struct{}

func (p *basicAudioProcessor) Process(ctx context.Context, file *MediaFile) (*AudioAnalysis, error) {
	metadata := map[string]interface{}{
		"size":      file.Size,
		"mime_type": file.MimeType,
	}

	/* Try to extract audio metadata using ffprobe if available */
	duration := 0.0
	sampleRate := 0
	channels := 0

	/* Check if we have file data or URL */
	filePath := file.URL
	if filePath == "" && len(file.Data) > 0 {
		/* Would need to write to temp file for ffprobe */
		/* For now, skip if no URL */
	}

	if filePath != "" {
		/* Try to get metadata using ffprobe */
		cmd := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-show_entries",
			"format=duration:stream=sample_rate,channels", "-of", "default=noprint_wrappers=1", filePath)
		output, err := cmd.CombinedOutput()
		if err == nil {
			/* Parse ffprobe output */
			outputStr := string(output)
			for _, line := range strings.Split(outputStr, "\n") {
				if strings.HasPrefix(line, "duration=") {
					if d, err := strconv.ParseFloat(strings.TrimPrefix(line, "duration="), 64); err == nil {
						duration = d
					}
				}
				if strings.HasPrefix(line, "sample_rate=") {
					if sr, err := strconv.Atoi(strings.TrimPrefix(line, "sample_rate=")); err == nil {
						sampleRate = sr
					}
				}
				if strings.HasPrefix(line, "channels=") {
					if ch, err := strconv.Atoi(strings.TrimPrefix(line, "channels=")); err == nil {
						channels = ch
					}
				}
			}
		}
	}

	metadata["duration"] = duration
	if sampleRate > 0 {
		metadata["sample_rate"] = sampleRate
	}
	if channels > 0 {
		metadata["channels"] = channels
	}

	/* Try transcription using Whisper if available */
	transcript := ""
	language := "unknown"

	/* Check if Whisper CLI is available */
	if filePath != "" {
		if _, err := exec.LookPath("whisper"); err == nil {
			/* Try to transcribe using Whisper CLI */
			cmd := exec.CommandContext(ctx, "whisper", filePath, "--output_format", "txt", "--language", "auto")
			_, err := cmd.CombinedOutput()
			if err == nil {
				/* Read transcript from output file */
				baseName := strings.TrimSuffix(filepath.Base(filePath), filepath.Ext(filePath))
				transcriptFile := baseName + ".txt"
				if data, err := os.ReadFile(transcriptFile); err == nil {
					transcript = strings.TrimSpace(string(data))
					os.Remove(transcriptFile) /* Clean up */
				}
			}
		}
	}

	/* For production, would integrate with:
	 * - OpenAI Whisper API
	 * - Google Speech-to-Text API
	 * - Azure Speech Services
	 * - AWS Transcribe
	 * - Local Whisper model
	 */
	if transcript == "" {
		if file.Metadata != nil {
			if preTranscribed, ok := file.Metadata["transcript"].(string); ok {
				transcript = preTranscribed
			}
		}
		if transcript == "" {
			transcript = "Audio transcription requires integration with speech-to-text service. Audio metadata extracted successfully."
		}
	}

	/* Detect language if transcript is available */
	if transcript != "" && language == "unknown" {
		/* Simple language detection could be added here */
		/* For now, assume English if transcript contains English characters */
		if strings.ContainsAny(transcript, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") {
			language = "en"
		}
	}

	return &AudioAnalysis{
		Transcript: transcript,
		Duration:   duration,
		Language:   "unknown", /* Would be detected by transcription service */
		Metadata:   metadata,
	}, nil
}

type basicVideoProcessor struct{}

func (p *basicVideoProcessor) Process(ctx context.Context, file *MediaFile) (*VideoAnalysis, error) {
	metadata := map[string]interface{}{
		"size":      file.Size,
		"mime_type": file.MimeType,
	}

	/* Try to extract video metadata using ffprobe if available */
	duration := 0.0
	width := 0
	height := 0
	fps := 0.0
	codec := "unknown"

	/* Check if we have file data or URL */
	filePath := file.URL
	if filePath == "" && len(file.Data) > 0 {
		/* Would need to write to temp file for ffprobe */
		/* For now, skip if no URL */
	}

	if filePath != "" {
		/* Try to get metadata using ffprobe */
		cmd := exec.CommandContext(ctx, "ffprobe", "-v", "error", "-show_entries",
			"format=duration:stream=width,height,r_frame_rate,codec_name", "-of", "default=noprint_wrappers=1", filePath)
		output, err := cmd.CombinedOutput()
		if err == nil {
			/* Parse ffprobe output */
			outputStr := string(output)
			for _, line := range strings.Split(outputStr, "\n") {
				if strings.HasPrefix(line, "duration=") {
					if d, err := strconv.ParseFloat(strings.TrimPrefix(line, "duration="), 64); err == nil {
						duration = d
					}
				}
				if strings.HasPrefix(line, "width=") {
					if w, err := strconv.Atoi(strings.TrimPrefix(line, "width=")); err == nil {
						width = w
					}
				}
				if strings.HasPrefix(line, "height=") {
					if h, err := strconv.Atoi(strings.TrimPrefix(line, "height=")); err == nil {
						height = h
					}
				}
				if strings.HasPrefix(line, "r_frame_rate=") {
					/* Parse frame rate (e.g., "30/1" or "25/1") */
					rateStr := strings.TrimPrefix(line, "r_frame_rate=")
					parts := strings.Split(rateStr, "/")
					if len(parts) == 2 {
						if num, err1 := strconv.ParseFloat(parts[0], 64); err1 == nil {
							if den, err2 := strconv.ParseFloat(parts[1], 64); err2 == nil && den > 0 {
								fps = num / den
							}
						}
					}
				}
				if strings.HasPrefix(line, "codec_name=") {
					codec = strings.TrimPrefix(line, "codec_name=")
				}
			}
		}
	}

	metadata["duration"] = duration
	if width > 0 {
		metadata["width"] = width
		metadata["height"] = height
		metadata["resolution"] = fmt.Sprintf("%dx%d", width, height)
	}
	if fps > 0 {
		metadata["fps"] = fps
	}
	if codec != "unknown" {
		metadata["codec"] = codec
	}

	/* For video analysis, would integrate with:
	 * - GPT-4 Vision (for frame analysis)
	 * - Video understanding models
	 * - Scene detection
	 * - Object tracking
	 */
	description := ""
	if file.Metadata != nil {
		if preAnalyzed, ok := file.Metadata["description"].(string); ok {
			description = preAnalyzed
		}
	}

	if description == "" {
		description = fmt.Sprintf("Video analysis requires integration with vision models. Video metadata extracted: duration=%.2fs, resolution=%dx%d, codec=%s",
			duration, width, height, codec)
	}

	return &VideoAnalysis{
		Description: description,
		Duration:    duration,
		Frames:      []FrameAnalysis{}, /* Would be populated by frame-by-frame analysis */
		Metadata:    metadata,
	}, nil
}
