# Notification Delivery (neurondb-cloud)

Production-ready notification delivery (email via SMTP, Slack). Copy into your `neurondb-cloud` repo and wire to your event bus or handlers.

## Configuration

| Env | Description |
|-----|-------------|
| SMTP_HOST | SMTP server host |
| SMTP_PORT | SMTP port (e.g. 587) |
| SMTP_USER | SMTP username |
| SMTP_PASSWORD | SMTP password |
| SMTP_FROM | From address (e.g. alerts@example.com) |
| SLACK_WEBHOOK_URL | Slack incoming webhook URL (optional) |

## Usage

```go
import "your-module/notifications/delivery"

// Email
err := delivery.SendEmail(ctx, delivery.Email{
    To: "user@example.com",
    Subject: "Backup completed",
    Body: "Your backup finished successfully.",
})

// Slack
err := delivery.SendSlack(ctx, delivery.SlackMessage{
    Text: "Backup failed for tenant xyz",
    Severity: "error",
})
```
