/*-------------------------------------------------------------------------
 *
 * time.go
 *    Time formatting and parsing utilities for NeuronAgent
 *
 * Provides time formatting functions using standard formats (ISO8601, RFC3339)
 * and time parsing utilities for consistent date/time handling.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/utils/time.go
 *
 *-------------------------------------------------------------------------
 */

package utils

import (
	"fmt"
	"time"
)

const (
	ISO8601Format  = "2006-01-02T15:04:05Z07:00"
	RFC3339Format  = time.RFC3339
	DateOnlyFormat = "2006-01-02"
	TimeOnlyFormat = "15:04:05"
	DateTimeFormat = "2006-01-02 15:04:05"
)

/* FormatTime formats time using ISO8601 format */
func FormatTime(t time.Time) string {
	return t.Format(ISO8601Format)
}

/* ParseTime parses time from ISO8601 format */
func ParseTime(s string) (time.Time, error) {
	return time.Parse(ISO8601Format, s)
}

/* FormatDuration formats duration as human-readable string */
func FormatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	if d < time.Minute {
		return fmt.Sprintf("%.2fs", d.Seconds())
	}
	if d < time.Hour {
		return fmt.Sprintf("%.2fm", d.Minutes())
	}
	return fmt.Sprintf("%.2fh", d.Hours())
}

/* ParseDuration parses duration string */
func ParseDuration(s string) (time.Duration, error) {
	return time.ParseDuration(s)
}

/* Now returns current time in UTC */
func Now() time.Time {
	return time.Now().UTC()
}

/* UnixTimestamp returns Unix timestamp */
func UnixTimestamp(t time.Time) int64 {
	return t.Unix()
}

/* FromUnixTimestamp creates time from Unix timestamp */
func FromUnixTimestamp(ts int64) time.Time {
	return time.Unix(ts, 0)
}

/* IsExpired checks if a time is before now */
func IsExpired(t time.Time) bool {
	return t.Before(time.Now())
}

/* TimeAgo returns human-readable time ago string */
func TimeAgo(t time.Time) string {
	duration := time.Since(t)

	if duration < time.Minute {
		return "just now"
	}
	if duration < time.Hour {
		minutes := int(duration.Minutes())
		if minutes == 1 {
			return "1 minute ago"
		}
		return fmt.Sprintf("%d minutes ago", minutes)
	}
	if duration < 24*time.Hour {
		hours := int(duration.Hours())
		if hours == 1 {
			return "1 hour ago"
		}
		return fmt.Sprintf("%d hours ago", hours)
	}

	days := int(duration.Hours() / 24)
	if days == 1 {
		return "1 day ago"
	}
	if days < 7 {
		return fmt.Sprintf("%d days ago", days)
	}

	weeks := days / 7
	if weeks == 1 {
		return "1 week ago"
	}
	if weeks < 4 {
		return fmt.Sprintf("%d weeks ago", weeks)
	}

	months := days / 30
	if months == 1 {
		return "1 month ago"
	}
	if months < 12 {
		return fmt.Sprintf("%d months ago", months)
	}

	years := days / 365
	if years == 1 {
		return "1 year ago"
	}
	return fmt.Sprintf("%d years ago", years)
}
