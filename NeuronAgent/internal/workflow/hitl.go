/*-------------------------------------------------------------------------
 *
 * hitl.go
 *    Human-in-the-loop integration for workflow engine
 *
 * Integrates human-in-the-loop approval steps with email/webhook notifications
 * using existing approval_requests table.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/workflow/hitl.go
 *
 *-------------------------------------------------------------------------
 */

package workflow

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
	"github.com/neurondb/NeuronAgent/internal/notifications"
)

type HITLManager struct {
	queries        *db.Queries
	emailService   *notifications.EmailService
	webhookService *notifications.WebhookService
	baseURL        string
}

func NewHITLManager(queries *db.Queries, emailService *notifications.EmailService, webhookService *notifications.WebhookService, baseURL string) *HITLManager {
	return &HITLManager{
		queries:        queries,
		emailService:   emailService,
		webhookService: webhookService,
		baseURL:        baseURL,
	}
}

/* SetEmailService sets the email service for notifications */
func (h *HITLManager) SetEmailService(service *notifications.EmailService) {
	h.emailService = service
}

/* SetWebhookService sets the webhook service for notifications */
func (h *HITLManager) SetWebhookService(service *notifications.WebhookService) {
	h.webhookService = service
}

/* SetBaseURL sets the base URL for approval links */
func (h *HITLManager) SetBaseURL(baseURL string) {
	h.baseURL = baseURL
}

/* ApprovalStepConfig defines configuration for an approval step */
type ApprovalStepConfig struct {
	ApprovalType   string                 `json:"approval_type"` // "email", "webhook", "ticket"
	Recipients     []string               `json:"recipients"`    // Email addresses or webhook URLs
	Subject        string                 `json:"subject"`
	Message        string                 `json:"message"`
	TimeoutSeconds int                    `json:"timeout_seconds"`
	Metadata       map[string]interface{} `json:"metadata"`
}

/* RequestApproval creates an approval request and sends notifications */
func (h *HITLManager) RequestApproval(ctx context.Context, workflowExecutionID uuid.UUID, stepExecutionID uuid.UUID, config ApprovalStepConfig) (*uuid.UUID, error) {
	/* Create approval request */
	approvalRequest := &db.ApprovalRequest{
		WorkflowExecutionID: &workflowExecutionID,
		StepExecutionID:     &stepExecutionID,
		ApprovalType:        config.ApprovalType,
		Status:              "pending",
		RequestedAt:         time.Now(),
		Metadata:            config.Metadata,
	}

	if err := h.queries.CreateApprovalRequest(ctx, approvalRequest); err != nil {
		return nil, fmt.Errorf("failed to create approval request: %w", err)
	}

	/* Send notifications based on type */
	switch config.ApprovalType {
	case "email":
		if err := h.sendEmailNotification(ctx, approvalRequest.ID, config); err != nil {
			/* Log error but don't fail - approval request is created */
			metrics.WarnWithContext(ctx, "Failed to send email notification for approval request", map[string]interface{}{
				"approval_id": approvalRequest.ID.String(),
				"error":       err.Error(),
			})
		}
	case "webhook":
		if err := h.sendWebhookNotification(ctx, approvalRequest.ID, config); err != nil {
			metrics.WarnWithContext(ctx, "Failed to send webhook notification for approval request", map[string]interface{}{
				"approval_id": approvalRequest.ID.String(),
				"error":       err.Error(),
			})
		}
	case "ticket":
		if err := h.createTicket(ctx, approvalRequest.ID, config); err != nil {
			metrics.WarnWithContext(ctx, "Failed to create ticket for approval request", map[string]interface{}{
				"approval_id": approvalRequest.ID.String(),
				"error":       err.Error(),
			})
		}
	}

	return &approvalRequest.ID, nil
}

/* WaitForApproval waits for approval decision */
func (h *HITLManager) WaitForApproval(ctx context.Context, approvalRequestID uuid.UUID, timeout time.Duration) (bool, error) {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case <-time.After(time.Until(deadline)):
			return false, fmt.Errorf("approval request timed out")
		case <-ticker.C:
			approval, err := h.queries.GetApprovalRequest(ctx, approvalRequestID)
			if err != nil {
				return false, fmt.Errorf("failed to get approval request: %w", err)
			}

			if approval.Status == "approved" {
				return true, nil
			}
			if approval.Status == "rejected" {
				return false, nil
			}
			/* Still pending, continue waiting */
		}
	}
}

/* sendEmailNotification sends email notification */
func (h *HITLManager) sendEmailNotification(ctx context.Context, approvalID uuid.UUID, config ApprovalStepConfig) error {
	if h.emailService == nil || !h.emailService.IsEnabled() {
		metrics.WarnWithContext(ctx, "Email service not configured, skipping email notification", map[string]interface{}{
			"approval_id": approvalID.String(),
		})
		return nil
	}

	if len(config.Recipients) == 0 {
		return fmt.Errorf("no email recipients specified")
	}

	/* Build email subject */
	subject := config.Subject
	if subject == "" {
		subject = fmt.Sprintf("Approval Request: %s", approvalID.String())
	}

	/* Build email body with approval/reject links */
	baseURL := h.baseURL
	if baseURL == "" {
		baseURL = "http://localhost:8080" /* Default fallback */
	}
	if metadata, ok := config.Metadata["base_url"].(string); ok {
		baseURL = metadata /* Metadata override takes precedence */
	}

	approveURL := fmt.Sprintf("%s/api/v1/approval-requests/%s/approve", baseURL, approvalID.String())
	rejectURL := fmt.Sprintf("%s/api/v1/approval-requests/%s/reject", baseURL, approvalID.String())

	body := config.Message
	if body == "" {
		body = fmt.Sprintf("An approval request has been created.\n\nApproval ID: %s\n\n", approvalID.String())
	}
	body += fmt.Sprintf("\n\nTo approve, visit: %s\nTo reject, visit: %s\n", approveURL, rejectURL)

	/* Send email to all recipients */
	var lastErr error
	for _, recipient := range config.Recipients {
		if err := h.emailService.SendEmail(ctx, recipient, subject, body); err != nil {
			metrics.WarnWithContext(ctx, "Failed to send email to recipient", map[string]interface{}{
				"approval_id": approvalID.String(),
				"recipient":   recipient,
				"error":       err.Error(),
			})
			lastErr = err
		} else {
			metrics.InfoWithContext(ctx, "Email notification sent successfully", map[string]interface{}{
				"approval_id": approvalID.String(),
				"recipient":   recipient,
			})
		}
	}

	return lastErr
}

/* sendWebhookNotification sends webhook notification */
func (h *HITLManager) sendWebhookNotification(ctx context.Context, approvalID uuid.UUID, config ApprovalStepConfig) error {
	if h.webhookService == nil {
		metrics.WarnWithContext(ctx, "Webhook service not configured, skipping webhook notification", map[string]interface{}{
			"approval_id": approvalID.String(),
		})
		return nil
	}

	if len(config.Recipients) == 0 {
		return fmt.Errorf("no webhook URLs specified")
	}

	/* Build webhook payload */
	baseURL := h.baseURL
	if baseURL == "" {
		baseURL = "http://localhost:8080" /* Default fallback */
	}
	if metadata, ok := config.Metadata["base_url"].(string); ok {
		baseURL = metadata /* Metadata override takes precedence */
	}

	approveURL := fmt.Sprintf("%s/api/v1/approval-requests/%s/approve", baseURL, approvalID.String())
	rejectURL := fmt.Sprintf("%s/api/v1/approval-requests/%s/reject", baseURL, approvalID.String())

	payload := map[string]interface{}{
		"approval_id":   approvalID.String(),
		"status":        "pending",
		"approval_type": config.ApprovalType,
		"subject":       config.Subject,
		"message":       config.Message,
		"approve_url":   approveURL,
		"reject_url":    rejectURL,
		"metadata":      config.Metadata,
		"created_at":   time.Now().Format(time.RFC3339),
	}

	/* Send webhook to all recipients */
	var lastErr error
	for _, url := range config.Recipients {
		if err := h.webhookService.SendWebhook(ctx, url, payload); err != nil {
			metrics.WarnWithContext(ctx, "Failed to send webhook to URL", map[string]interface{}{
				"approval_id": approvalID.String(),
				"url":         url,
				"error":       err.Error(),
			})
			lastErr = err
		} else {
			metrics.InfoWithContext(ctx, "Webhook notification sent successfully", map[string]interface{}{
				"approval_id": approvalID.String(),
				"url":         url,
			})
		}
	}

	return lastErr
}

/* createTicket creates a ticket in external system */
func (h *HITLManager) createTicket(ctx context.Context, approvalID uuid.UUID, config ApprovalStepConfig) error {
	/* Extract ticket system configuration from metadata */
	ticketSystem := "jira" /* Default */
	if system, ok := config.Metadata["ticket_system"].(string); ok {
		ticketSystem = strings.ToLower(system)
	}

	baseURL := ""
	if url, ok := config.Metadata["ticket_base_url"].(string); ok {
		baseURL = url
	}

	apiKey := ""
	if key, ok := config.Metadata["ticket_api_key"].(string); ok {
		apiKey = key
	}

	projectKey := ""
	if key, ok := config.Metadata["ticket_project_key"].(string); ok {
		projectKey = key
	}

	/* Enhanced validation for ticket system */
	if baseURL == "" {
		return fmt.Errorf("ticket system configuration incomplete: base_url is required")
	}
	if apiKey == "" {
		return fmt.Errorf("ticket system configuration incomplete: api_key is required")
	}

	/* Validate base URL format */
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		return fmt.Errorf("ticket system base_url must start with http:// or https://")
	}

	/* Validate ticket system type */
	validSystems := []string{"jira", "servicenow", "zendesk", "github", "gitlab", "linear"}
	isValid := false
	for _, valid := range validSystems {
		if ticketSystem == valid {
			isValid = true
			break
		}
	}
	if !isValid {
		return fmt.Errorf("unsupported ticket system: %s (supported: %v)", ticketSystem, validSystems)
	}

	/* Build ticket payload based on system type */
	var ticketPayload map[string]interface{}
	var endpoint string

	switch ticketSystem {
	case "jira":
		endpoint = fmt.Sprintf("%s/rest/api/3/issue", baseURL)
		ticketPayload = map[string]interface{}{
			"fields": map[string]interface{}{
				"project": map[string]interface{}{
					"key": projectKey,
				},
				"summary":     config.Subject,
				"description": config.Message,
				"issuetype": map[string]interface{}{
					"name": "Task",
				},
			},
		}
	case "servicenow":
		endpoint = fmt.Sprintf("%s/api/now/table/incident", baseURL)
		ticketPayload = map[string]interface{}{
			"short_description": config.Subject,
			"description":       config.Message,
		}
	case "zendesk":
		endpoint = fmt.Sprintf("%s/api/v2/tickets.json", baseURL)
		ticketPayload = map[string]interface{}{
			"ticket": map[string]interface{}{
				"subject": config.Subject,
				"comment": map[string]interface{}{
					"body": config.Message,
				},
			},
		}
	case "github":
		endpoint = fmt.Sprintf("%s/repos/%s/issues", baseURL, projectKey)
		ticketPayload = map[string]interface{}{
			"title": config.Subject,
			"body":  config.Message,
		}
	case "gitlab":
		endpoint = fmt.Sprintf("%s/api/v4/projects/%s/issues", baseURL, projectKey)
		ticketPayload = map[string]interface{}{
			"title":       config.Subject,
			"description": config.Message,
		}
	case "linear":
		endpoint = fmt.Sprintf("%s/graphql", baseURL)
		/* Linear uses GraphQL */
		ticketPayload = map[string]interface{}{
			"query": fmt.Sprintf(`
				mutation {
					issueCreate(
						input: {
							title: "%s"
							description: "%s"
							teamId: "%s"
						}
					) {
						success
						issue { id }
					}
				}
			`, config.Subject, config.Message, projectKey),
		}
	default:
		/* Generic REST API */
		endpoint = fmt.Sprintf("%s/api/tickets", baseURL)
		ticketPayload = map[string]interface{}{
			"title":       config.Subject,
			"description": config.Message,
			"approval_id": approvalID.String(),
		}
	}

	/* Create HTTP request */
	payloadJSON, err := json.Marshal(ticketPayload)
	if err != nil {
		return fmt.Errorf("failed to marshal ticket payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, strings.NewReader(string(payloadJSON)))
	if err != nil {
		return fmt.Errorf("failed to create ticket request: %w", err)
	}

	/* Set headers */
	req.Header.Set("Content-Type", "application/json")
	switch ticketSystem {
	case "jira":
		req.Header.Set("Authorization", fmt.Sprintf("Basic %s", apiKey))
	case "zendesk":
		req.Header.Set("Authorization", fmt.Sprintf("Basic %s", base64.StdEncoding.EncodeToString([]byte(apiKey+":x"))))
	case "github", "gitlab":
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	case "linear":
		req.Header.Set("Authorization", apiKey)
		req.Header.Set("Content-Type", "application/json")
	default:
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
	}

	/* Send request with retry logic */
	client := &http.Client{
		Timeout: 30 * time.Second,
	}
	
	var resp *http.Response
	maxRetries := 3
	retryDelay := 1 * time.Second
	
	for attempt := 0; attempt < maxRetries; attempt++ {
		resp, err = client.Do(req)
		if err == nil && resp.StatusCode >= 200 && resp.StatusCode < 300 {
			break /* Success */
		}
		
		if attempt < maxRetries-1 {
			time.Sleep(retryDelay)
			retryDelay *= 2 /* Exponential backoff */
			/* Recreate request for retry */
			req, _ = http.NewRequestWithContext(ctx, "POST", endpoint, strings.NewReader(string(payloadJSON)))
			/* Re-set headers */
			switch ticketSystem {
			case "jira":
				req.Header.Set("Authorization", fmt.Sprintf("Basic %s", apiKey))
			case "zendesk":
				req.Header.Set("Authorization", fmt.Sprintf("Basic %s", base64.StdEncoding.EncodeToString([]byte(apiKey+":x"))))
			case "github", "gitlab":
				req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
			case "linear":
				req.Header.Set("Authorization", apiKey)
			default:
				req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", apiKey))
			}
			req.Header.Set("Content-Type", "application/json")
		}
	}
	
	if err != nil {
		return fmt.Errorf("ticket creation request failed after %d attempts: %w", maxRetries, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("ticket creation failed: status_code=%d, body=%s", resp.StatusCode, string(body))
	}

	/* Parse response to get ticket ID */
	var responseBody map[string]interface{}
	decoder := json.NewDecoder(resp.Body)
	if err := decoder.Decode(&responseBody); err == nil {
		var ticketID string
		switch ticketSystem {
		case "jira":
			if key, ok := responseBody["key"].(string); ok {
				ticketID = key
			}
		case "servicenow":
			if result, ok := responseBody["result"].(map[string]interface{}); ok {
				if sysID, ok := result["sys_id"].(string); ok {
					ticketID = sysID
				}
			}
		case "zendesk":
			if ticket, ok := responseBody["ticket"].(map[string]interface{}); ok {
				if id, ok := ticket["id"].(float64); ok {
					ticketID = fmt.Sprintf("%.0f", id)
				}
			}
		case "github":
			if number, ok := responseBody["number"].(float64); ok {
				ticketID = fmt.Sprintf("%.0f", number)
			}
		case "gitlab":
			if iid, ok := responseBody["iid"].(float64); ok {
				ticketID = fmt.Sprintf("%.0f", iid)
			}
		case "linear":
			if data, ok := responseBody["data"].(map[string]interface{}); ok {
				if issueCreate, ok := data["issueCreate"].(map[string]interface{}); ok {
					if issue, ok := issueCreate["issue"].(map[string]interface{}); ok {
						if id, ok := issue["id"].(string); ok {
							ticketID = id
						}
					}
				}
			}
		default:
			if id, ok := responseBody["id"].(string); ok {
				ticketID = id
			} else if id, ok := responseBody["id"].(float64); ok {
				ticketID = fmt.Sprintf("%.0f", id)
			}
		}

		if ticketID != "" {
			metrics.InfoWithContext(ctx, "Ticket created successfully", map[string]interface{}{
				"approval_id": approvalID.String(),
				"ticket_id":   ticketID,
				"system":      ticketSystem,
			})

			/* Store ticket ID in approval request metadata */
			if config.Metadata == nil {
				config.Metadata = make(map[string]interface{})
			}
			config.Metadata["ticket_id"] = ticketID
		}
	}

	return nil
}

/* ExecuteApprovalStep executes an approval step, blocking until approval */
func (h *HITLManager) ExecuteApprovalStep(ctx context.Context, workflowExecutionID uuid.UUID, stepExecutionID uuid.UUID, step *db.WorkflowStep, inputs map[string]interface{}) (map[string]interface{}, error) {
	/* Extract approval config from step inputs */
	config := ApprovalStepConfig{
		ApprovalType:   "email",
		TimeoutSeconds: 3600, /* 1 hour default */
	}

	/* Extract approval config from step inputs */
	/* step.Inputs is JSONBMap which is map[string]interface{} */
	if step.Inputs != nil {
		if configMap, ok := step.Inputs["approval_config"].(map[string]interface{}); ok {
		if approvalType, ok := configMap["approval_type"].(string); ok {
			config.ApprovalType = approvalType
		}
		if recipients, ok := configMap["recipients"].([]interface{}); ok {
			config.Recipients = make([]string, len(recipients))
			for i, r := range recipients {
				if str, ok := r.(string); ok {
					config.Recipients[i] = str
				}
			}
		}
		if subject, ok := configMap["subject"].(string); ok {
			config.Subject = subject
		}
		if message, ok := configMap["message"].(string); ok {
			config.Message = message
		}
		if timeout, ok := configMap["timeout_seconds"].(float64); ok {
			config.TimeoutSeconds = int(timeout)
		}
		if metadata, ok := configMap["metadata"].(map[string]interface{}); ok {
			config.Metadata = metadata
		}
		}
	}

	/* Request approval */
	approvalID, err := h.RequestApproval(ctx, workflowExecutionID, stepExecutionID, config)
	if err != nil {
		return nil, fmt.Errorf("failed to request approval: %w", err)
	}

	/* Wait for approval */
	timeout := time.Duration(config.TimeoutSeconds) * time.Second
	approved, err := h.WaitForApproval(ctx, *approvalID, timeout)
	if err != nil {
		return nil, fmt.Errorf("approval wait failed: %w", err)
	}

	if !approved {
		return nil, fmt.Errorf("approval was rejected")
	}

	/* Return approval result */
	outputs := map[string]interface{}{
		"approved":    true,
		"approval_id": approvalID.String(),
		"approved_at": time.Now(),
	}

	return outputs, nil
}
