package session

import (
	"context"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
)

type CleanupService struct {
	queries   *db.Queries
	interval  time.Duration
	maxAge    time.Duration
	ctx       context.Context
	cancel    context.CancelFunc
	done      chan struct{}
}

func NewCleanupService(queries *db.Queries, interval, maxAge time.Duration) *CleanupService {
	ctx, cancel := context.WithCancel(context.Background())
	return &CleanupService{
		queries:  queries,
		interval: interval,
		maxAge:   maxAge,
		ctx:      ctx,
		cancel:   cancel,
		done:     make(chan struct{}),
	}
}

// Start starts the cleanup service
func (s *CleanupService) Start() {
	go s.run()
}

// Stop stops the cleanup service
func (s *CleanupService) Stop() {
	s.cancel()
	<-s.done
}

func (s *CleanupService) run() {
	defer close(s.done)

	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	// Run immediately on start
	s.cleanup()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.cleanup()
		}
	}
}

func (s *CleanupService) cleanup() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Delete sessions older than maxAge
	cutoffTime := time.Now().Add(-s.maxAge)
	
	// Get all agents to check their sessions
	agents, err := s.queries.ListAgents(ctx)
	if err != nil {
		return
	}

	for _, agent := range agents {
		sessions, err := s.queries.ListSessions(ctx, agent.ID, 1000, 0)
		if err != nil {
			continue
		}

		for _, session := range sessions {
			if session.LastActivityAt.Before(cutoffTime) {
				_ = s.queries.DeleteSession(ctx, session.ID)
			}
		}
	}
}

