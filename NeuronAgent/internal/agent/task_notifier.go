/*-------------------------------------------------------------------------
 *
 * task_notifier.go
 *    Task notification system for completion alerts
 *
 * Sends notifications when async tasks complete, fail, or reach milestones.
 * Supports email, webhook, and push notification channels.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/task_notifier.go
 *
 *-------------------------------------------------------------------------
 */

package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/notifications"
)

/* TaskNotifier manages task completion notifications */
type TaskNotifier struct {
	queries        *db.Queries
	emailService   *notifications.EmailService
	webhookService *notifications.WebhookService
}

/* NewTaskNotifier creates a new task notifier */
func NewTaskNotifier(queries *db.Queries, emailService *notifications.EmailService, webhookService *notifications.WebhookService) *TaskNotifier {
	return &TaskNotifier{
		queries:        queries,
		emailService:   emailService,
		webhookService: webhookService,
	}
}

/* SendCompletionAlert sends a completion notification for a task */
func (n *TaskNotifier) SendCompletionAlert(ctx context.Context, taskID uuid.UUID) error {
	/* Get task details */
	task, err := n.getTaskDetails(ctx, taskID)
	if err != nil {
		return fmt.Errorf("completion alert failed: task_retrieval_error=true, task_id='%s', error=%w", taskID.String(), err)
	}

	/* Get alert preferences */
	prefs, err := n.getAlertPreferences(ctx, task.AgentID, nil)
	if err != nil || len(prefs) == 0 {
		/* No preferences found, skip notification */
		return nil
	}

	/* Send notifications via configured channels */
	for _, pref := range prefs {
		if !pref.Enabled {
			continue
		}

		/* Check if completion alerts are enabled */
		hasCompletion := false
		for _, alertType := range pref.AlertTypes {
			if alertType == "completion" {
				hasCompletion = true
				break
			}
		}
		if !hasCompletion {
			continue
		}

		/* Send via each enabled channel */
		for _, channel := range pref.Channels {
			switch channel {
			case "email":
				if pref.EmailAddress != nil && *pref.EmailAddress != "" {
					n.sendEmailNotification(ctx, taskID, "completion", *pref.EmailAddress, task)
				}
			case "webhook":
				if pref.WebhookURL != nil && *pref.WebhookURL != "" {
					n.sendWebhookNotification(ctx, taskID, "completion", *pref.WebhookURL, task)
				}
			}
		}
	}

	return nil
}

/* SendFailureAlert sends a failure notification for a task */
func (n *TaskNotifier) SendFailureAlert(ctx context.Context, taskID uuid.UUID, taskErr error) error {
	/* Get task details */
	task, err := n.getTaskDetails(ctx, taskID)
	if err != nil {
		return fmt.Errorf("failure alert failed: task_retrieval_error=true, task_id='%s', error=%w", taskID.String(), err)
	}

	/* Get alert preferences */
	prefs, err := n.getAlertPreferences(ctx, task.AgentID, nil)
	if err != nil || len(prefs) == 0 {
		return nil
	}

	/* Send notifications */
	for _, pref := range prefs {
		if !pref.Enabled {
			continue
		}

		hasFailure := false
		for _, alertType := range pref.AlertTypes {
			if alertType == "failure" {
				hasFailure = true
				break
			}
		}
		if !hasFailure {
			continue
		}

		for _, channel := range pref.Channels {
			errorMsg := ""
			if taskErr != nil {
				errorMsg = taskErr.Error()
			}

			switch channel {
			case "email":
				if pref.EmailAddress != nil && *pref.EmailAddress != "" {
					n.sendEmailNotification(ctx, taskID, "failure", *pref.EmailAddress, task, errorMsg)
				}
			case "webhook":
				if pref.WebhookURL != nil && *pref.WebhookURL != "" {
					n.sendWebhookNotification(ctx, taskID, "failure", *pref.WebhookURL, task, errorMsg)
				}
			}
		}
	}

	return nil
}

/* SendProgressAlert sends a progress notification for a task */
func (n *TaskNotifier) SendProgressAlert(ctx context.Context, taskID uuid.UUID, progress float64, message string) error {
	/* Get task details */
	task, err := n.getTaskDetails(ctx, taskID)
	if err != nil {
		return fmt.Errorf("progress alert failed: task_retrieval_error=true, task_id='%s', error=%w", taskID.String(), err)
	}

	/* Get alert preferences */
	prefs, err := n.getAlertPreferences(ctx, task.AgentID, nil)
	if err != nil || len(prefs) == 0 {
		return nil
	}

	/* Send notifications */
	for _, pref := range prefs {
		if !pref.Enabled {
			continue
		}

		hasProgress := false
		for _, alertType := range pref.AlertTypes {
			if alertType == "progress" {
				hasProgress = true
				break
			}
		}
		if !hasProgress {
			continue
		}

		for _, channel := range pref.Channels {
			switch channel {
			case "email":
				if pref.EmailAddress != nil && *pref.EmailAddress != "" {
					n.sendEmailNotification(ctx, taskID, "progress", *pref.EmailAddress, task, fmt.Sprintf("Progress: %.1f%% - %s", progress*100, message))
				}
			case "webhook":
				if pref.WebhookURL != nil && *pref.WebhookURL != "" {
					n.sendWebhookNotification(ctx, taskID, "progress", *pref.WebhookURL, task, fmt.Sprintf("Progress: %.1f%% - %s", progress*100, message))
				}
			}
		}
	}

	return nil
}

/* getTaskDetails retrieves task details for notification */
func (n *TaskNotifier) getTaskDetails(ctx context.Context, taskID uuid.UUID) (*AsyncTask, error) {
	query := `SELECT id, session_id, agent_id, task_type, status, priority, input, result, error_message,
		created_at, started_at, completed_at, metadata
		FROM neurondb_agent.async_tasks
		WHERE id = $1`

	task := &AsyncTask{}
	var inputJSON, resultJSON []byte
	var errorMsg *string

	err := n.queries.DB.QueryRowContext(ctx, query, taskID).Scan(
		&task.ID, &task.SessionID, &task.AgentID, &task.TaskType, &task.Status, &task.Priority,
		&inputJSON, &resultJSON, &errorMsg,
		&task.CreatedAt, &task.StartedAt, &task.CompletedAt, &task.Metadata,
	)

	if err != nil {
		return nil, err
	}

	return task, nil
}

/* AlertPreference represents user alert preferences */
type AlertPreference struct {
	ID           uuid.UUID
	UserID       *uuid.UUID
	AgentID      *uuid.UUID
	AlertTypes   []string
	Channels     []string
	EmailAddress *string
	WebhookURL   *string
	Enabled      bool
}

/* getAlertPreferences retrieves alert preferences */
func (n *TaskNotifier) getAlertPreferences(ctx context.Context, agentID uuid.UUID, userID *uuid.UUID) ([]*AlertPreference, error) {
	query := `SELECT id, user_id, agent_id, alert_types, channels, email_address, webhook_url, enabled
		FROM neurondb_agent.task_alert_preferences
		WHERE enabled = true AND (agent_id = $1 OR agent_id IS NULL)`
	args := []interface{}{agentID}

	if userID != nil {
		query += " AND (user_id = $2 OR user_id IS NULL)"
		args = append(args, *userID)
	}

	rows, err := n.queries.DB.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var prefs []*AlertPreference
	for rows.Next() {
		pref := &AlertPreference{}
		err := rows.Scan(
			&pref.ID, &pref.UserID, &pref.AgentID,
			&pref.AlertTypes, &pref.Channels,
			&pref.EmailAddress, &pref.WebhookURL, &pref.Enabled,
		)
		if err != nil {
			continue
		}
		prefs = append(prefs, pref)
	}

	return prefs, nil
}

/* sendEmailNotification sends an email notification */
func (n *TaskNotifier) sendEmailNotification(ctx context.Context, taskID uuid.UUID, alertType, recipient string, task *AsyncTask, extraData ...string) {
	if n.emailService == nil {
		return
	}

	/* Create alert record */
	alertID := uuid.New()
	query := `INSERT INTO neurondb_agent.task_alerts
		(id, task_id, alert_type, channel, recipient, status, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)`

	_, err := n.queries.DB.ExecContext(ctx, query,
		alertID, taskID, alertType, "email", recipient, "pending", time.Now(),
	)
	if err != nil {
		return
	}

	/* Send email */
	subject := fmt.Sprintf("Task %s: %s", alertType, task.TaskType)
	body := fmt.Sprintf("Task ID: %s\nType: %s\nStatus: %s", task.ID.String(), task.TaskType, task.Status)
	if len(extraData) > 0 {
		body += "\n" + extraData[0]
	}

	err = n.emailService.SendEmail(ctx, recipient, subject, body)
	if err != nil {
		/* Update alert status to failed */
		updateQuery := `UPDATE neurondb_agent.task_alerts
			SET status = 'failed', error_message = $1
			WHERE id = $2`
		errMsg := err.Error()
		n.queries.DB.ExecContext(ctx, updateQuery, errMsg, alertID)
		return
	}

	/* Update alert status to sent */
	updateQuery := `UPDATE neurondb_agent.task_alerts
		SET status = 'sent', sent_at = NOW()
		WHERE id = $1`
	n.queries.DB.ExecContext(ctx, updateQuery, alertID)
}

/* sendWebhookNotification sends a webhook notification */
func (n *TaskNotifier) sendWebhookNotification(ctx context.Context, taskID uuid.UUID, alertType, webhookURL string, task *AsyncTask, extraData ...string) {
	if n.webhookService == nil {
		return
	}

	/* Create alert record */
	alertID := uuid.New()
	query := `INSERT INTO neurondb_agent.task_alerts
		(id, task_id, alert_type, channel, recipient, status, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)`

	_, err := n.queries.DB.ExecContext(ctx, query,
		alertID, taskID, alertType, "webhook", webhookURL, "pending", time.Now(),
	)
	if err != nil {
		return
	}

	/* Prepare payload */
	payload := map[string]interface{}{
		"task_id":    task.ID.String(),
		"alert_type": alertType,
		"task_type":  task.TaskType,
		"status":     task.Status,
	}
	if len(extraData) > 0 {
		payload["message"] = extraData[0]
	}

	/* Send webhook */
	err = n.webhookService.SendWebhook(ctx, webhookURL, payload)
	if err != nil {
		/* Update alert status to failed */
		updateQuery := `UPDATE neurondb_agent.task_alerts
			SET status = 'failed', error_message = $1
			WHERE id = $2`
		errMsg := err.Error()
		n.queries.DB.ExecContext(ctx, updateQuery, errMsg, alertID)
		return
	}

	/* Update alert status to sent */
	updateQuery := `UPDATE neurondb_agent.task_alerts
		SET status = 'sent', sent_at = NOW()
		WHERE id = $1`
	n.queries.DB.ExecContext(ctx, updateQuery, alertID)
}
