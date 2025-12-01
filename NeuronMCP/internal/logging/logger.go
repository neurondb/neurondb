package logging

import (
	"io"
	"os"
	"time"

	"github.com/neurondb/NeuronMCP/internal/config"
	"github.com/rs/zerolog"
)

// Logger provides structured logging
type Logger struct {
	logger zerolog.Logger
	level  zerolog.Level
}

// NewLogger creates a new logger
func NewLogger(cfg *config.LoggingConfig) *Logger {
	// Parse level
	var level zerolog.Level
	switch cfg.Level {
	case "debug":
		level = zerolog.DebugLevel
	case "info":
		level = zerolog.InfoLevel
	case "warn":
		level = zerolog.WarnLevel
	case "error":
		level = zerolog.ErrorLevel
	default:
		level = zerolog.InfoLevel
	}

	// Determine output
	var output io.Writer
	if cfg.Output != nil {
		switch *cfg.Output {
		case "stdout":
			output = os.Stdout
		case "stderr":
			output = os.Stderr
		default:
			// Try to open as file
			if file, err := os.OpenFile(*cfg.Output, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644); err == nil {
				output = file
			} else {
				output = os.Stderr
			}
		}
	} else {
		output = os.Stderr
	}

	// Configure format
	if cfg.Format == "json" {
		zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	} else {
		output = zerolog.ConsoleWriter{Out: output, TimeFormat: time.RFC3339}
	}

	logger := zerolog.New(output).With().Timestamp().Logger().Level(level)

	return &Logger{
		logger: logger,
		level:  level,
	}
}

// Debug logs a debug message
func (l *Logger) Debug(message string, metadata map[string]interface{}) {
	l.log(zerolog.DebugLevel, message, metadata)
}

// Info logs an info message
func (l *Logger) Info(message string, metadata map[string]interface{}) {
	l.log(zerolog.InfoLevel, message, metadata)
}

// Warn logs a warning message
func (l *Logger) Warn(message string, metadata map[string]interface{}) {
	l.log(zerolog.WarnLevel, message, metadata)
}

// Error logs an error message
func (l *Logger) Error(message string, err error, metadata map[string]interface{}) {
	event := l.logger.Error()
	if err != nil {
		event = event.Err(err)
	}
	if metadata != nil {
		event = event.Fields(metadata)
	}
	event.Msg(message)
}

func (l *Logger) log(level zerolog.Level, message string, metadata map[string]interface{}) {
	if level < l.level {
		return
	}

	event := l.logger.WithLevel(level)
	if metadata != nil {
		event = event.Fields(metadata)
	}
	event.Msg(message)
}

// Child creates a child logger with additional metadata
func (l *Logger) Child(metadata map[string]interface{}) *Logger {
	childLogger := l.logger.With().Fields(metadata).Logger()
	return &Logger{
		logger: childLogger,
		level:  l.level,
	}
}

