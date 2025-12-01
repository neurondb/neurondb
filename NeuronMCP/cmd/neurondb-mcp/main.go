package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"github.com/neurondb/NeuronMCP/internal/server"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		cancel()
	}()

	// Create and start server
	srv, err := server.NewServer()
	if err != nil {
		os.Stderr.WriteString("Failed to create server: " + err.Error() + "\n")
		os.Exit(1)
	}

	// Start server
	if err := srv.Start(ctx); err != nil {
		if err != context.Canceled {
			os.Stderr.WriteString("Server error: " + err.Error() + "\n")
			os.Exit(1)
		}
	}

	// Cleanup
	if err := srv.Stop(); err != nil {
		os.Stderr.WriteString("Error stopping server: " + err.Error() + "\n")
	}
}

