/*-------------------------------------------------------------------------
 *
 * memory_pubsub.go
 *    Distributed memory pub-sub for memory synchronization
 *
 * Provides pub-sub mechanism for memory updates across cluster nodes
 * using PostgreSQL LISTEN/NOTIFY.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/distributed/memory_pubsub.go
 *
 *-------------------------------------------------------------------------
 */

package distributed

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* MemoryPubSub manages memory synchronization across nodes */
type MemoryPubSub struct {
	queries    *db.Queries
	listener   *pq.Listener
	subscribers map[string][]MemorySubscriber
	mu         sync.RWMutex
	ctx        context.Context
	cancel     context.CancelFunc
}

/* MemorySubscriber is a callback for memory events */
type MemorySubscriber func(ctx context.Context, event MemoryEvent) error

/* MemoryEvent represents a memory update event */
type MemoryEvent struct {
	Type      string    `json:"type"`      // "create", "update", "delete"
	SessionID uuid.UUID `json:"session_id"`
	AgentID   uuid.UUID `json:"agent_id"`
	ChunkID   uuid.UUID `json:"chunk_id,omitempty"`
	Content   string    `json:"content,omitempty"`
}

/* NewMemoryPubSub creates a new memory pub-sub instance */
func NewMemoryPubSub(queries *db.Queries) *MemoryPubSub {
	ctx, cancel := context.WithCancel(context.Background())
	return &MemoryPubSub{
		queries:     queries,
		subscribers: make(map[string][]MemorySubscriber),
		ctx:         ctx,
		cancel:      cancel,
	}
}

/* Start starts the pub-sub listener */
func (mps *MemoryPubSub) Start(ctx context.Context) error {
	/* Get connection string from environment or construct from DB connection */
	/* For LISTEN/NOTIFY, we need a persistent connection string */
	connStr := os.Getenv("DATABASE_URL")
	if connStr == "" {
		/* Construct from environment variables */
		host := os.Getenv("DB_HOST")
		if host == "" {
			host = "localhost"
		}
		port := getEnvInt("DB_PORT", 5432)
		user := os.Getenv("DB_USER")
		if user == "" {
			user = "neurondb"
		}
		password := os.Getenv("DB_PASSWORD")
		if password == "" {
			password = "neurondb"
		}
		dbname := os.Getenv("DB_NAME")
		if dbname == "" {
			dbname = "neurondb"
		}
		
		connStr = fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
			host, port, user, password, dbname)
	}
	
	/* Create PostgreSQL listener */
	reportProblem := func(ev pq.ListenerEventType, err error) {
		if err != nil {
			metrics.WarnWithContext(ctx, "Memory pub-sub listener error", map[string]interface{}{
				"event": int(ev),
				"error": err.Error(),
			})
		}
	}

	listener := pq.NewListener(connStr, 10*time.Second, time.Minute, reportProblem)
	
	/* Listen to memory channel */
	err := listener.Listen("neurondb_agent_memory")
	if err != nil {
		return fmt.Errorf("memory pub-sub start failed: listen_error=true, error=%w", err)
	}

	mps.listener = listener

	/* Start event processing */
	go mps.processEvents(mps.ctx)

	return nil
}

/* getEnvInt gets integer from environment variable with default */
func getEnvInt(key string, defaultValue int) int {
	if val := os.Getenv(key); val != "" {
		if intVal, err := strconv.Atoi(val); err == nil {
			return intVal
		}
	}
	return defaultValue
}

/* Stop stops the pub-sub listener */
func (mps *MemoryPubSub) Stop(ctx context.Context) {
	if mps.listener != nil {
		mps.listener.Close()
	}
	mps.cancel()
}

/* processEvents processes incoming events */
func (mps *MemoryPubSub) processEvents(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case notification := <-mps.listener.Notify:
			if notification == nil {
				continue
			}

			/* Parse event */
			event, err := mps.parseEvent(notification.Extra)
			if err != nil {
				metrics.WarnWithContext(ctx, "Failed to parse memory event", map[string]interface{}{
					"error": err.Error(),
				})
				continue
			}

			/* Notify subscribers */
			mps.notifySubscribers(ctx, event)
		}
	}
}

/* parseEvent parses a notification payload into a MemoryEvent */
func (mps *MemoryPubSub) parseEvent(payload string) (*MemoryEvent, error) {
	/* Parse JSON payload */
	var event MemoryEvent
	if err := json.Unmarshal([]byte(payload), &event); err != nil {
		/* Fallback to basic event if parsing fails */
		return &MemoryEvent{
			Type: "update",
		}, nil
	}
	
	return &event, nil
}

/* notifySubscribers notifies all subscribers of an event */
func (mps *MemoryPubSub) notifySubscribers(ctx context.Context, event *MemoryEvent) {
	mps.mu.RLock()
	subscribers := mps.subscribers[event.Type]
	mps.mu.RUnlock()

	for _, subscriber := range subscribers {
		if err := subscriber(ctx, *event); err != nil {
			metrics.WarnWithContext(ctx, "Subscriber error", map[string]interface{}{
				"event_type": event.Type,
				"error":      err.Error(),
			})
		}
	}
}

/* Subscribe subscribes to memory events */
func (mps *MemoryPubSub) Subscribe(eventType string, subscriber MemorySubscriber) {
	mps.mu.Lock()
	defer mps.mu.Unlock()

	mps.subscribers[eventType] = append(mps.subscribers[eventType], subscriber)
}

/* Publish publishes a memory event */
func (mps *MemoryPubSub) Publish(ctx context.Context, event MemoryEvent) error {
	/* Publish via PostgreSQL NOTIFY */
	query := `SELECT pg_notify('neurondb_agent_memory', $1)`
	
	/* Serialize event to JSON */
	payload := fmt.Sprintf(`{"type":"%s","session_id":"%s","agent_id":"%s"}`, 
		event.Type, event.SessionID.String(), event.AgentID.String())
	
	_, err := mps.queries.DB.ExecContext(ctx, query, payload)
	if err != nil {
		return fmt.Errorf("memory publish failed: notify_error=true, error=%w", err)
	}

	return nil
}

