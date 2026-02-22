package logging

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"
)

/* Logger provides structured logging */
type Logger struct {
	level  string
	format string
	output *os.File
}

/* NewLogger creates a new logger */
func NewLogger(level, format, output string) *Logger {
	var file *os.File
	var err error

	if output != "stdout" && output != "stderr" {
		file, err = os.OpenFile(output, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Printf("Failed to open log file %s: %v, using stdout", output, err)
			file = os.Stdout
		}
	} else if output == "stderr" {
		file = os.Stderr
	} else {
		file = os.Stdout
	}

	return &Logger{
		level:  level,
		format: format,
		output: file,
	}
}

/* LogEntry represents a log entry */
type LogEntry struct {
	Timestamp string                 `json:"timestamp"`
	Level     string                 `json:"level"`
	Message   string                 `json:"message"`
	Fields    map[string]interface{} `json:"fields,omitempty"`
}

func (l *Logger) shouldLog(level string) bool {
	levels := map[string]int{
		"debug": 0,
		"info":  1,
		"warn":  2,
		"error": 3,
	}

	currentLevel, ok := levels[l.level]
	if !ok {
		currentLevel = 1 /* Default to info */
	}

	logLevel, ok := levels[level]
	if !ok {
		return true
	}

	return logLevel >= currentLevel
}

func (l *Logger) log(level, message string, fields map[string]interface{}) {
	if !l.shouldLog(level) {
		return
	}

	entry := LogEntry{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Level:     level,
		Message:   message,
		Fields:    fields,
	}

	if l.format == "json" {
		data, _ := json.Marshal(entry)
		fmt.Fprintln(l.output, string(data))
	} else {
		fieldStr := ""
		if len(fields) > 0 {
			fieldStr = fmt.Sprintf(" %+v", fields)
		}
		fmt.Fprintf(l.output, "[%s] %s: %s%s\n", entry.Timestamp, level, message, fieldStr)
	}
}

/* Debug logs a debug message */
func (l *Logger) Debug(message string, fields map[string]interface{}) {
	l.log("debug", message, fields)
}

/* Info logs an info message */
func (l *Logger) Info(message string, fields map[string]interface{}) {
	l.log("info", message, fields)
}

/* Warn logs a warning message */
func (l *Logger) Warn(message string, fields map[string]interface{}) {
	l.log("warn", message, fields)
}

/* Error logs an error message */
func (l *Logger) Error(message string, err error, fields map[string]interface{}) {
	if fields == nil {
		fields = make(map[string]interface{})
	}
	if err != nil {
		fields["error"] = err.Error()
	}
	l.log("error", message, fields)
}

/* Close releases file handles when logging to a file. Call on shutdown for clean exit.
 * No-op when output is stdout or stderr. */
func (l *Logger) Close() error {
	if l.output != nil && l.output != os.Stdout && l.output != os.Stderr {
		return l.output.Close()
	}
	return nil
}
