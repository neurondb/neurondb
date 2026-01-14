/*-------------------------------------------------------------------------
 *
 * logging.go
 *    Logging initialization and configuration for NeuronAgent
 *
 * Provides logging setup using zerolog with configurable levels and formats.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/metrics/logging.go
 *
 *-------------------------------------------------------------------------
 */

package metrics

import (
	"os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func InitLogging(level string, format string) {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix

	/* Set log level */
	switch level {
	case "debug":
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	case "info":
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	case "warn":
		zerolog.SetGlobalLevel(zerolog.WarnLevel)
	case "error":
		zerolog.SetGlobalLevel(zerolog.ErrorLevel)
	default:
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}

	/* Set format */
	if format == "console" {
		log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	} else {
		log.Logger = log.Output(os.Stderr)
	}
}

func Logger() *zerolog.Logger {
	return &log.Logger
}
