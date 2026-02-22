package handlers

import (
	"sync"
	"time"
)

/* IngestJobStatus represents a single ingest job's status */
type IngestJobStatus struct {
	JobID        string    `json:"job_id"`
	ProfileID    string    `json:"profile_id"`
	Status       string    `json:"status"` /* "queued", "running", "completed", "failed" */
	Progress     int       `json:"progress"`
	RowsIngested int64     `json:"rows_ingested,omitempty"`
	TableName    string    `json:"table_name,omitempty"`
	Error        string    `json:"error,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}

/* IngestJobStore persists ingest job status for GetIngestStatus and ListIngestJobs */
type IngestJobStore interface {
	SetQueued(profileID, jobID, tableName string)
	SetRunning(profileID, jobID string)
	SetCompleted(profileID, jobID string, rowsIngested int64)
	SetFailed(profileID, jobID string, err error)
	Get(profileID, jobID string) (*IngestJobStatus, bool)
	List(profileID string) []IngestJobStatus
}

/* MemoryIngestStore is an in-memory implementation of IngestJobStore */
type MemoryIngestStore struct {
	mu   sync.RWMutex
	jobs map[string]*IngestJobStatus
}

/* NewMemoryIngestStore creates a new in-memory ingest job store */
func NewMemoryIngestStore() *MemoryIngestStore {
	return &MemoryIngestStore{jobs: make(map[string]*IngestJobStatus)}
}

func (s *MemoryIngestStore) key(profileID, jobID string) string {
	return profileID + "\x00" + jobID
}

func (s *MemoryIngestStore) SetQueued(profileID, jobID, tableName string) {
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[s.key(profileID, jobID)] = &IngestJobStatus{
		JobID: jobID, ProfileID: profileID, Status: "queued", Progress: 0, TableName: tableName,
		CreatedAt: now, UpdatedAt: now,
	}
}

func (s *MemoryIngestStore) SetRunning(profileID, jobID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if j, ok := s.jobs[s.key(profileID, jobID)]; ok {
		j.Status = "running"
		j.Progress = 50
		j.UpdatedAt = time.Now()
	}
}

func (s *MemoryIngestStore) SetCompleted(profileID, jobID string, rowsIngested int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if j, ok := s.jobs[s.key(profileID, jobID)]; ok {
		j.Status = "completed"
		j.Progress = 100
		j.RowsIngested = rowsIngested
		j.UpdatedAt = time.Now()
	}
}

func (s *MemoryIngestStore) SetFailed(profileID, jobID string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	errStr := ""
	if err != nil {
		errStr = err.Error()
	}
	if j, ok := s.jobs[s.key(profileID, jobID)]; ok {
		j.Status = "failed"
		j.Error = errStr
		j.UpdatedAt = time.Now()
	}
}

func (s *MemoryIngestStore) Get(profileID, jobID string) (*IngestJobStatus, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	j, ok := s.jobs[s.key(profileID, jobID)]
	if !ok || j == nil {
		return nil, false
	}
	cp := *j
	return &cp, true
}

func (s *MemoryIngestStore) List(profileID string) []IngestJobStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()
	prefix := profileID + "\x00"
	var out []IngestJobStatus
	for k, j := range s.jobs {
		if j == nil {
			continue
		}
		if len(k) >= len(prefix) && k[:len(prefix)] == prefix {
			out = append(out, *j)
		}
	}
	return out
}
