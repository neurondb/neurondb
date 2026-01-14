/*-------------------------------------------------------------------------
 *
 * scheduler.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/jobs/scheduler.go
 *
 *-------------------------------------------------------------------------
 */

package jobs

import (
	"context"
	"sync"
	"time"
)

type ScheduledJob struct {
	ID       string
	CronExpr string
	JobType  string
	Payload  map[string]interface{}
	NextRun  time.Time
	Enabled  bool
}

type Scheduler struct {
	queue  *Queue
	jobs   map[string]*ScheduledJob
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
	ticker *time.Ticker
}

func NewScheduler(queue *Queue) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())
	return &Scheduler{
		queue:  queue,
		jobs:   make(map[string]*ScheduledJob),
		ctx:    ctx,
		cancel: cancel,
		ticker: time.NewTicker(1 * time.Minute), /* Check every minute */
	}
}

/* Start starts the scheduler */
func (s *Scheduler) Start() {
	s.wg.Add(1)
	go s.run()
}

/* Stop stops the scheduler */
func (s *Scheduler) Stop() {
	s.cancel()
	s.ticker.Stop()
	s.wg.Wait()
}

func (s *Scheduler) run() {
	defer s.wg.Done()

	/* Check immediately */
	s.checkAndRun()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-s.ticker.C:
			s.checkAndRun()
		}
	}
}

func (s *Scheduler) checkAndRun() {
	s.mu.RLock()
	now := time.Now()
	var jobsToRun []*ScheduledJob

	for _, job := range s.jobs {
		if job.Enabled && !job.NextRun.After(now) {
			jobsToRun = append(jobsToRun, job)
		}
	}
	s.mu.RUnlock()

	/* Run jobs */
	for _, job := range jobsToRun {
		s.runJob(job)
	}
}

func (s *Scheduler) runJob(job *ScheduledJob) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	/* Enqueue job */
	_, err := s.queue.Enqueue(ctx, job.JobType, nil, nil, job.Payload, 0)
	if err != nil {
		return
	}

	/* Calculate next run time (simplified - in production use cron parser) */
	s.mu.Lock()
	job.NextRun = time.Now().Add(1 * time.Hour) /* Default: run every hour */
	s.mu.Unlock()
}

/* Schedule adds a scheduled job */
func (s *Scheduler) Schedule(id, cronExpr, jobType string, payload map[string]interface{}) error {
	/* Parse cron expression (simplified - in production use robfig/cron) */
	/* For now, parse simple patterns like "0 * * * *" (every hour) */
	nextRun := parseCronExpression(cronExpr)

	s.mu.Lock()
	s.jobs[id] = &ScheduledJob{
		ID:       id,
		CronExpr: cronExpr,
		JobType:  jobType,
		Payload:  payload,
		NextRun:  nextRun,
		Enabled:  true,
	}
	s.mu.Unlock()

	return nil
}

/* parseCronExpression parses a simple cron expression (simplified) */
func parseCronExpression(expr string) time.Time {
	/* Default: run every hour */
	/* In production, use a proper cron parser like robfig/cron */
	return time.Now().Add(1 * time.Hour)
}

/* Unschedule removes a scheduled job */
func (s *Scheduler) Unschedule(id string) {
	s.mu.Lock()
	delete(s.jobs, id)
	s.mu.Unlock()
}

/* List returns all scheduled jobs */
func (s *Scheduler) List() []*ScheduledJob {
	s.mu.RLock()
	defer s.mu.RUnlock()

	jobs := make([]*ScheduledJob, 0, len(s.jobs))
	for _, job := range s.jobs {
		jobs = append(jobs, job)
	}
	return jobs
}
