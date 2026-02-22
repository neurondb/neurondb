package main

import (
	"context"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"metering/pkg/collector"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		panic("DATABASE_URL is required")
	}
	windowMin := 60
	if s := os.Getenv("METERING_WINDOW_MINUTES"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			windowMin = n
		}
	}
	dryRun := os.Getenv("METERING_DRY_RUN") != ""

	cfg := collector.Config{
		DatabaseURL:   dbURL,
		WindowMinutes: windowMin,
		DryRun:        dryRun,
	}
	c, err := collector.New(ctx, cfg)
	if err != nil {
		panic(err)
	}
	defer c.Close()

	// Run once; for cron use: run and exit. For daemon use a ticker.
	_, err = c.Run(ctx)
	if err != nil {
		panic(err)
	}

	// Optional: run in loop (e.g. METERING_DAEMON=1)
	if os.Getenv("METERING_DAEMON") == "1" {
		ticker := time.NewTicker(time.Duration(windowMin) * time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				_, _ = c.Run(ctx)
			}
		}
	}
}
