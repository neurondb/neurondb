# Notification Delivery (neuron-cloud)

This document describes how to implement notification delivery (email, Slack) for **neuron-cloud** so that users and admins receive alerts and lifecycle notifications.

## Goals

- Send email (e.g. backup success/failure, billing, alerts).
- Send Slack notifications (e.g. incident alerts, approval requests).
- Use real credentials (no placeholders) in production.

## Suggested approach

### 1. Notification service

In the control plane, add or extend a **notification** service that:

- Subscribes to events (e.g. from an event bus or DB polling): backup completed, backup failed, budget exceeded, approval requested.
- Renders a message (template) per channel type (email vs Slack).
- Calls a **delivery** layer that sends via SMTP (email) or Slack Webhook/API.

### 2. Email (SMTP)

- Config: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`.
- Use a standard Go SMTP client (e.g. `net/smtp` or a small library).
- Templates: plain text and/or HTML for “Backup completed”, “Backup failed”, “Billing summary”, etc.

### 3. Slack

- Config: `SLACK_WEBHOOK_URL` (per channel or default) or `SLACK_BOT_TOKEN` for API.
- On event: POST to webhook or call `chat.postMessage` with a short JSON payload (title, text, severity).

### 4. Alertmanager (Prometheus)

- Replace placeholder SMTP/Slack/PagerDuty settings in the observability stack with real credentials (from env or secrets).
- Ensure `alertmanager.yml` uses variables that are injected at deploy time (e.g. from Kubernetes secrets).

### 5. API (optional)

- `POST /api/v1/notifications/preferences` – user/org notification preferences (email on/off, Slack channel).
- Store preferences in the control plane DB and have the notification service respect them when sending.

When the **neuron-cloud** repo is present, implement the notification service and wire it to the event source and delivery (SMTP, Slack); then replace Alertmanager placeholders with real configuration.
