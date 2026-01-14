/*-------------------------------------------------------------------------
 *
 * email.go
 *    Email notification service
 *
 * Provides SMTP-based email notifications for task alerts and system events.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/notifications/email.go
 *
 *-------------------------------------------------------------------------
 */

package notifications

import (
	"context"
	"fmt"
	"net/smtp"
	"strings"
)

/* EmailService provides email notification capabilities */
type EmailService struct {
	smtpHost     string
	smtpPort     int
	smtpUser     string
	smtpPassword string
	smtpFrom     string
	enabled      bool
}

/* NewEmailService creates a new email service */
func NewEmailService(smtpHost string, smtpPort int, smtpUser, smtpPassword, smtpFrom string) *EmailService {
	return &EmailService{
		smtpHost:     smtpHost,
		smtpPort:     smtpPort,
		smtpUser:     smtpUser,
		smtpPassword: smtpPassword,
		smtpFrom:     smtpFrom,
		enabled:      smtpHost != "" && smtpPort > 0,
	}
}

/* SendEmail sends an email notification */
func (e *EmailService) SendEmail(ctx context.Context, to, subject, body string) error {
	if !e.enabled {
		return fmt.Errorf("email service not configured")
	}

	/* Validate email address */
	if !strings.Contains(to, "@") {
		return fmt.Errorf("invalid email address: %s", to)
	}

	/* Prepare message */
	msg := fmt.Sprintf("From: %s\r\n", e.smtpFrom)
	msg += fmt.Sprintf("To: %s\r\n", to)
	msg += fmt.Sprintf("Subject: %s\r\n", subject)
	msg += "\r\n"
	msg += body

	/* SMTP authentication */
	auth := smtp.PlainAuth("", e.smtpUser, e.smtpPassword, e.smtpHost)

	/* Send email */
	addr := fmt.Sprintf("%s:%d", e.smtpHost, e.smtpPort)
	err := smtp.SendMail(addr, auth, e.smtpFrom, []string{to}, []byte(msg))
	if err != nil {
		return fmt.Errorf("email send failed: to='%s', subject='%s', error=%w", to, subject, err)
	}

	return nil
}

/* SendHTMLEmail sends an HTML email notification */
func (e *EmailService) SendHTMLEmail(ctx context.Context, to, subject, htmlBody string) error {
	if !e.enabled {
		return fmt.Errorf("email service not configured")
	}

	/* Validate email address */
	if !strings.Contains(to, "@") {
		return fmt.Errorf("invalid email address: %s", to)
	}

	/* Prepare message with HTML content */
	msg := fmt.Sprintf("From: %s\r\n", e.smtpFrom)
	msg += fmt.Sprintf("To: %s\r\n", to)
	msg += fmt.Sprintf("Subject: %s\r\n", subject)
	msg += "MIME-Version: 1.0\r\n"
	msg += "Content-Type: text/html; charset=UTF-8\r\n"
	msg += "\r\n"
	msg += htmlBody

	/* SMTP authentication */
	auth := smtp.PlainAuth("", e.smtpUser, e.smtpPassword, e.smtpHost)

	/* Send email */
	addr := fmt.Sprintf("%s:%d", e.smtpHost, e.smtpPort)
	err := smtp.SendMail(addr, auth, e.smtpFrom, []string{to}, []byte(msg))
	if err != nil {
		return fmt.Errorf("html email send failed: to='%s', subject='%s', error=%w", to, subject, err)
	}

	return nil
}

/* IsEnabled returns whether email service is enabled */
func (e *EmailService) IsEnabled() bool {
	return e.enabled
}
