package delivery

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/smtp"
	"os"
	"strconv"
)

// Email holds a single email to send.
type Email struct {
	To      []string
	Subject string
	Body    string
	IsHTML  bool
}

// SendEmail sends an email via SMTP using env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM.
func SendEmail(ctx context.Context, e Email) error {
	host := os.Getenv("SMTP_HOST")
	if host == "" {
		return fmt.Errorf("SMTP_HOST is required")
	}
	port := 587
	if s := os.Getenv("SMTP_PORT"); s != "" {
		if p, err := strconv.Atoi(s); err == nil {
			port = p
		}
	}
	user := os.Getenv("SMTP_USER")
	pass := os.Getenv("SMTP_PASSWORD")
	from := os.Getenv("SMTP_FROM")
	if from == "" {
		from = user
	}
	addr := fmt.Sprintf("%s:%d", host, port)
	auth := smtp.PlainAuth("", user, pass, host)
	contentType := "text/plain; charset=utf-8"
	if e.IsHTML {
		contentType = "text/html; charset=utf-8"
	}
	msg := []byte(
		"From: " + from + "\r\n" +
			"To: " + e.To[0] + "\r\n" +
			"Subject: " + e.Subject + "\r\n" +
			"Content-Type: " + contentType + "\r\n" +
			"\r\n" + e.Body + "\r\n")
	if len(e.To) == 0 {
		return fmt.Errorf("no recipient")
	}
	if port == 465 {
		tlsConfig := &tls.Config{ServerName: host}
		conn, err := tls.Dial("tcp", addr, tlsConfig)
		if err != nil {
			return err
		}
		defer conn.Close()
		client, err := smtp.NewClient(conn, host)
		if err != nil {
			return err
		}
		defer client.Close()
		if auth != nil {
			if err := client.Auth(auth); err != nil {
				return err
			}
		}
		if err := client.Mail(from); err != nil {
			return err
		}
		for _, to := range e.To {
			if err := client.Rcpt(to); err != nil {
				return err
			}
		}
		w, err := client.Data()
		if err != nil {
			return err
		}
		_, err = w.Write(msg)
		if err != nil {
			return err
		}
		return w.Close()
	}
	return smtp.SendMail(addr, auth, from, e.To, msg)
}
