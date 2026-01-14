/*-------------------------------------------------------------------------
 *
 * cleanup.go
 *    Session cleanup service for NeuronAgent
 *
 * Provides background service for automatically cleaning up expired
 * sessions based on configurable age and interval settings.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/session/cleanup.go
 *
 *-------------------------------------------------------------------------
 */

package session

import (
	"context"
	"fmt"
	"time"

	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

type CleanupService struct {
	queries  *db.Queries
	interval time.Duration
	maxAge   time.Duration
	ctx      context.Context
	cancel   context.CancelFunc
	done     chan struct{}
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

/* Start starts the cleanup service */
func (s *CleanupService) Start() {
	go s.run()
}

/* Stop stops the cleanup service */
func (s *CleanupService) Stop() {
	s.cancel()
	<-s.done
}

func (s *CleanupService) run() {
	defer close(s.done)

	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	/* Run immediately on start */
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

	/* Recover from panics in cleanup */
	defer func() {
		if r := recover(); r != nil {
			metrics.ErrorWithContext(ctx, "Panic in session cleanup", fmt.Errorf("panic: %v", r), nil)
		}
	}()

	/* Delete sessions older than maxAge */
	cutoffTime := time.Now().Add(-s.maxAge)

	/* Get all agents to check their sessions */
	agents, err := s.queries.ListAgents(ctx)
	if err != nil {
		/* Log error but don't crash - cleanup will retry on next interval */
		metrics.WarnWithContext(ctx, "Failed to list agents for session cleanup", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	for _, agent := range agents {
		/* Check context cancellation */
		if ctx.Err() != nil {
			return
		}

		sessions, err := s.queries.ListSessions(ctx, agent.ID, 1000, 0)
		if err != nil {
			/* Log error but continue with other agents */
			metrics.WarnWithContext(ctx, "Failed to list sessions for cleanup", map[string]interface{}{
				"agent_id": agent.ID.String(),
				"error":    err.Error(),
			})
			continue
		}

		for _, session := range sessions {
			/* Check context cancellation */
			if ctx.Err() != nil {
				return
			}

			if session.LastActivityAt.Before(cutoffTime) {
				if err := s.queries.DeleteSession(ctx, session.ID); err != nil {
					/* Log deletion error but continue with other sessions */
					metrics.WarnWithContext(ctx, "Failed to delete expired session", map[string]interface{}{
						"session_id": session.ID.String(),
						"agent_id":   agent.ID.String(),
						"error":      err.Error(),
					})
				}
			}
		}
	}
}
