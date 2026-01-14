/*-------------------------------------------------------------------------
 *
 * moderation.go
 *    Content moderation and safety guardrails
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/safety/moderation.go
 *
 *-------------------------------------------------------------------------
 */

package safety

import (
	"context"
	"fmt"
	"regexp"
	"strings"
)

/* ModerationResult represents the result of content moderation */
type ModerationResult struct {
	Allowed    bool
	Reason     string
	Confidence float64
	Issues     []string
}

/* ContentModerator performs content moderation */
type ContentModerator struct {
	enablePIIDetection  bool
	enableToxicityCheck bool
	piiPatterns         []*regexp.Regexp
	toxicPatterns       []*regexp.Regexp
}

/* NewContentModerator creates a new content moderator */
func NewContentModerator(enablePIIDetection, enableToxicityCheck bool) *ContentModerator {
	cm := &ContentModerator{
		enablePIIDetection:  enablePIIDetection,
		enableToxicityCheck: enableToxicityCheck,
	}

	/* Initialize PII patterns */
	if enablePIIDetection {
		cm.piiPatterns = []*regexp.Regexp{
			/* Email */
			regexp.MustCompile(`\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b`),
			/* Phone (US format) */
			regexp.MustCompile(`\b\d{3}-\d{3}-\d{4}\b`),
			/* SSN (US format) */
			regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`),
			/* Credit card */
			regexp.MustCompile(`\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`),
			/* IP address */
			regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`),
		}
	}

	/* Initialize toxic patterns (simplified - in production would use ML model) */
	if enableToxicityCheck {
		toxicKeywords := []string{
			"kill", "death", "violence", "hate", "discrimination",
			/* Add more patterns as needed */
		}
		cm.toxicPatterns = make([]*regexp.Regexp, len(toxicKeywords))
		for i, keyword := range toxicKeywords {
			cm.toxicPatterns[i] = regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(keyword) + `\b`)
		}
	}

	return cm
}

/* ModerateInput moderates input content */
func (cm *ContentModerator) ModerateInput(ctx context.Context, content string) (*ModerationResult, error) {
	result := &ModerationResult{
		Allowed:    true,
		Confidence: 1.0,
		Issues:     []string{},
	}

	if content == "" {
		return result, nil
	}

	/* Check for PII */
	if cm.enablePIIDetection {
		piiFound := cm.detectPII(content)
		if len(piiFound) > 0 {
			result.Allowed = false
			result.Reason = "PII detected"
			result.Issues = append(result.Issues, fmt.Sprintf("PII detected: %v", piiFound))
			result.Confidence = 0.9
		}
	}

	/* Check for toxicity */
	if cm.enableToxicityCheck {
		toxicFound := cm.detectToxicity(content)
		if len(toxicFound) > 0 {
			result.Allowed = false
			result.Reason = "Toxic content detected"
			result.Issues = append(result.Issues, fmt.Sprintf("Toxic patterns: %v", toxicFound))
			result.Confidence = 0.8
		}
	}

	return result, nil
}

/* ModerateOutput moderates output content */
func (cm *ContentModerator) ModerateOutput(ctx context.Context, content string) (*ModerationResult, error) {
	/* Similar to ModerateInput but may have different rules */
	return cm.ModerateInput(ctx, content)
}

/* RedactPII redacts PII from content */
func (cm *ContentModerator) RedactPII(ctx context.Context, content string) (string, []string) {
	if !cm.enablePIIDetection {
		return content, []string{}
	}

	redacted := content
	var detected []string

	for _, pattern := range cm.piiPatterns {
		matches := pattern.FindAllString(content, -1)
		if len(matches) > 0 {
			detected = append(detected, matches...)
			redacted = pattern.ReplaceAllString(redacted, "[REDACTED]")
		}
	}

	return redacted, detected
}

/* detectPII detects PII in content */
func (cm *ContentModerator) detectPII(content string) []string {
	var found []string
	for _, pattern := range cm.piiPatterns {
		matches := pattern.FindAllString(content, -1)
		found = append(found, matches...)
	}
	return found
}

/* detectToxicity detects toxic content */
func (cm *ContentModerator) detectToxicity(content string) []string {
	var found []string
	contentLower := strings.ToLower(content)
	for _, pattern := range cm.toxicPatterns {
		matches := pattern.FindAllString(contentLower, -1)
		if len(matches) > 0 {
			found = append(found, matches...)
		}
	}
	return found
}

/* IsSafe checks if content is safe */
func (cm *ContentModerator) IsSafe(ctx context.Context, content string) (bool, error) {
	result, err := cm.ModerateInput(ctx, content)
	if err != nil {
		return false, err
	}
	return result.Allowed, nil
}
