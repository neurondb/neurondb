/*-------------------------------------------------------------------------
 *
 * progress.go
 *    Progress tracking for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/progress/progress.go
 *
 *-------------------------------------------------------------------------
 */

package progress

import (
	"fmt"
	"regexp"
	"sync"
	"time"
)

const (
	DefaultMaxProgressItems = 1000
	DefaultCleanupInterval  = 5 * time.Minute
	DefaultMaxAge           = 1 * time.Hour
	MaxProgressIDLength     = 200
)

var (
	progressIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
)

/* ProgressStatus represents progress status */
type ProgressStatus struct {
	ID        string                 `json:"id"`
	Status    string                 `json:"status"` /* pending, running, completed, failed */
	Progress  float64                `json:"progress"` /* 0.0 to 1.0 */
	Message   string                 `json:"message,omitempty"`
	StartedAt time.Time              `json:"startedAt"`
	UpdatedAt time.Time              `json:"updatedAt"`
	CompletedAt *time.Time           `json:"completedAt,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

/* Tracker tracks progress for operations */
type Tracker struct {
	progresses     map[string]*ProgressStatus
	mu             sync.RWMutex
	maxItems       int
	cleanupTicker  *time.Ticker
	cleanupStop    chan struct{}
	cleanupRunning bool
	cleanupMu      sync.Mutex
}

/* NewTracker creates a new progress tracker */
func NewTracker() *Tracker {
	t := &Tracker{
		progresses: make(map[string]*ProgressStatus),
		maxItems:   DefaultMaxProgressItems,
		cleanupStop: make(chan struct{}),
	}
	
	/* Start automatic cleanup */
	t.startCleanup()
	
	return t
}

/* startCleanup starts the automatic cleanup goroutine */
func (t *Tracker) startCleanup() {
	t.cleanupMu.Lock()
	defer t.cleanupMu.Unlock()
	
	if t.cleanupRunning {
		return
	}
	
	t.cleanupTicker = time.NewTicker(DefaultCleanupInterval)
	t.cleanupRunning = true
	
	go func() {
		for {
			select {
			case <-t.cleanupTicker.C:
				t.Cleanup(DefaultMaxAge)
			case <-t.cleanupStop:
				return
			}
		}
	}()
}

/* StopCleanup stops the automatic cleanup */
func (t *Tracker) StopCleanup() {
	t.cleanupMu.Lock()
	defer t.cleanupMu.Unlock()
	
	if !t.cleanupRunning {
		return
	}
	
	if t.cleanupTicker != nil {
		t.cleanupTicker.Stop()
	}
	close(t.cleanupStop)
	t.cleanupRunning = false
}

/* SetMaxItems sets the maximum number of tracked items */
func (t *Tracker) SetMaxItems(max int) {
	if max < 1 {
		max = DefaultMaxProgressItems
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.maxItems = max
}

/* validateProgressID validates a progress ID format */
func validateProgressID(id string) error {
	if id == "" {
		return fmt.Errorf("progress ID cannot be empty")
	}
	if len(id) > MaxProgressIDLength {
		return fmt.Errorf("progress ID too long: id_length=%d, max_length=%d", len(id), MaxProgressIDLength)
	}
	if !progressIDPattern.MatchString(id) {
		return fmt.Errorf("progress ID contains invalid characters: id='%s' (must be alphanumeric, underscore, or hyphen)", id)
	}
	return nil
}

/* Start starts tracking progress */
func (t *Tracker) Start(id string, message string) (*ProgressStatus, error) {
	if t == nil {
		return nil, fmt.Errorf("progress tracker instance is nil")
	}

	/* Validate progress ID */
	if err := validateProgressID(id); err != nil {
		return nil, fmt.Errorf("progress: invalid progress ID: %w", err)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	/* Check if we're at capacity */
	if len(t.progresses) >= t.maxItems {
		/* Try to clean up old entries first */
		now := time.Now()
		for pid, p := range t.progresses {
			if p.CompletedAt != nil && now.Sub(*p.CompletedAt) > DefaultMaxAge {
				delete(t.progresses, pid)
			}
		}
		
		/* If still at capacity, return error */
		if len(t.progresses) >= t.maxItems {
			return nil, fmt.Errorf("progress: maximum number of tracked items reached: current_count=%d, max_items=%d", len(t.progresses), t.maxItems)
		}
	}

	now := time.Now()
	progress := &ProgressStatus{
		ID:        id,
		Status:    "running",
		Progress:  0.0,
		Message:   message,
		StartedAt: now,
		UpdatedAt: now,
		Metadata:  make(map[string]interface{}),
	}

	t.progresses[id] = progress
	return progress, nil
}

/* Update updates progress */
func (t *Tracker) Update(id string, progress float64, message string) error {
	if t == nil {
		return fmt.Errorf("progress tracker instance is nil")
	}

	/* Validate progress ID */
	if err := validateProgressID(id); err != nil {
		return fmt.Errorf("progress: invalid progress ID: %w", err)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	p, exists := t.progresses[id]
	if !exists {
		return fmt.Errorf("progress: progress not found: id='%s'", id)
	}

	if progress < 0.0 {
		progress = 0.0
	}
	if progress > 1.0 {
		progress = 1.0
	}

	p.Progress = progress
	if message != "" {
		p.Message = message
	}
	p.UpdatedAt = time.Now()

	return nil
}

/* Complete marks progress as completed */
func (t *Tracker) Complete(id string, message string) error {
	if t == nil {
		return fmt.Errorf("progress tracker instance is nil")
	}

	/* Validate progress ID */
	if err := validateProgressID(id); err != nil {
		return fmt.Errorf("progress: invalid progress ID: %w", err)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	p, exists := t.progresses[id]
	if !exists {
		return fmt.Errorf("progress: progress not found: id='%s'", id)
	}

	now := time.Now()
	p.Status = "completed"
	p.Progress = 1.0
	if message != "" {
		p.Message = message
	}
	p.UpdatedAt = now
	p.CompletedAt = &now

	return nil
}

/* Fail marks progress as failed */
func (t *Tracker) Fail(id string, err error) error {
	if t == nil {
		return fmt.Errorf("progress tracker instance is nil")
	}

	/* Validate progress ID */
	if err := validateProgressID(id); err != nil {
		return fmt.Errorf("progress: invalid progress ID: %w", err)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	p, exists := t.progresses[id]
	if !exists {
		return fmt.Errorf("progress: progress not found: id='%s'", id)
	}

	now := time.Now()
	p.Status = "failed"
	p.UpdatedAt = now
	p.CompletedAt = &now
	if err != nil {
		p.Error = err.Error()
	}

	return nil
}

/* Get gets progress status */
func (t *Tracker) Get(id string) (*ProgressStatus, error) {
	if t == nil {
		return nil, fmt.Errorf("progress tracker instance is nil")
	}

	/* Validate progress ID */
	if err := validateProgressID(id); err != nil {
		return nil, fmt.Errorf("progress: invalid progress ID: %w", err)
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	p, exists := t.progresses[id]
	if !exists {
		return nil, fmt.Errorf("progress: progress not found: id='%s'", id)
	}

	return p, nil
}

/* List lists all progress statuses */
func (t *Tracker) List() []*ProgressStatus {
	t.mu.RLock()
	defer t.mu.RUnlock()

	statuses := make([]*ProgressStatus, 0, len(t.progresses))
	for _, p := range t.progresses {
		statuses = append(statuses, p)
	}

	return statuses
}

/* Cleanup removes old completed progress entries */
func (t *Tracker) Cleanup(maxAge time.Duration) {
	if t == nil {
		return
	}

	if maxAge < 0 {
		maxAge = DefaultMaxAge
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	now := time.Now()
	removedCount := 0
	for id, p := range t.progresses {
		if p.CompletedAt != nil {
			if now.Sub(*p.CompletedAt) > maxAge {
				delete(t.progresses, id)
				removedCount++
			}
		}
	}

	/* Also enforce max items limit by removing oldest completed entries */
	if len(t.progresses) > t.maxItems {
		/* Sort by completion time and remove oldest */
		type entry struct {
			id        string
			completed time.Time
		}
		var entries []entry
		for id, p := range t.progresses {
			if p.CompletedAt != nil {
				entries = append(entries, entry{id: id, completed: *p.CompletedAt})
			}
		}
		
		/* Remove oldest entries until under limit */
		toRemove := len(t.progresses) - t.maxItems
		for i := 0; i < toRemove && i < len(entries); i++ {
			delete(t.progresses, entries[i].id)
			removedCount++
		}
	}
}

