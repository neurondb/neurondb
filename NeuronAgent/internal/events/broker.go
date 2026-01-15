/*-------------------------------------------------------------------------
 *
 * broker.go
 *    Event-driven architecture broker
 *
 * Provides event streaming for agent actions, tool calls, memory updates
 * with support for Kafka/NATS and event sourcing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/events/broker.go
 *
 *-------------------------------------------------------------------------
 */

package events

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/neurondb/NeuronAgent/internal/db"
	"github.com/neurondb/NeuronAgent/internal/metrics"
)

/* Broker manages event streaming */
type Broker struct {
	queries     *db.Queries
	backends    []EventBackend
	subscribers map[string][]EventSubscriber
	mu          sync.RWMutex
	enabled     bool
}

/* EventBackend interface for event backends */
type EventBackend interface {
	Publish(ctx context.Context, topic string, event Event) error
	Subscribe(ctx context.Context, topic string, handler EventHandler) error
	Close() error
}

/* EventSubscriber is a callback for events */
type EventSubscriber func(ctx context.Context, event Event) error

/* EventHandler handles events from backends */
type EventHandler func(ctx context.Context, event Event) error

/* Event represents an event */
type Event struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	Source    string                 `json:"source"`
	Data      map[string]interface{} `json:"data"`
}

/* EventType represents event types */
type EventType string

const (
	EventTypeAgentAction    EventType = "agent.action"
	EventTypeToolCall       EventType = "tool.call"
	EventTypeToolResult     EventType = "tool.result"
	EventTypeMemoryUpdate   EventType = "memory.update"
	EventTypeMemoryRetrieve EventType = "memory.retrieve"
	EventTypeSessionStart   EventType = "session.start"
	EventTypeSessionEnd     EventType = "session.end"
	EventTypeError          EventType = "error"
)

/* NewBroker creates a new event broker */
func NewBroker(queries *db.Queries) *Broker {
	return &Broker{
		queries:     queries,
		backends:    make([]EventBackend, 0),
		subscribers: make(map[string][]EventSubscriber),
		enabled:     false,
	}
}

/* Enable enables event streaming */
func (b *Broker) Enable(ctx context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.enabled {
		return nil
	}

	/* Add PostgreSQL backend by default */
	pgBackend := NewPostgreSQLBackend(b.queries)
	b.backends = append(b.backends, pgBackend)

	b.enabled = true
	metrics.InfoWithContext(ctx, "Event broker enabled", map[string]interface{}{
		"backends": len(b.backends),
	})

	return nil
}

/* Disable disables event streaming */
func (b *Broker) Disable(ctx context.Context) {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, backend := range b.backends {
		backend.Close()
	}

	b.backends = make([]EventBackend, 0)
	b.enabled = false
}

/* Publish publishes an event */
func (b *Broker) Publish(ctx context.Context, eventType EventType, source string, data map[string]interface{}) error {
	if !b.enabled {
		return nil
	}

	event := Event{
		ID:        uuid.New().String(),
		Type:      string(eventType),
		Timestamp: time.Now(),
		Source:    source,
		Data:      data,
	}

	/* Publish to all backends */
	var lastErr error
	for _, backend := range b.backends {
		if err := backend.Publish(ctx, string(eventType), event); err != nil {
			lastErr = err
			metrics.WarnWithContext(ctx, "Failed to publish event", map[string]interface{}{
				"event_type": eventType,
				"error":      err.Error(),
			})
		}
	}

	/* Store in database for event sourcing */
	if err := b.storeEvent(ctx, event); err != nil {
		metrics.WarnWithContext(ctx, "Failed to store event", map[string]interface{}{
			"event_type": eventType,
			"error":      err.Error(),
		})
	}

	/* Notify local subscribers */
	b.notifySubscribers(ctx, event)

	return lastErr
}

/* Subscribe subscribes to events */
func (b *Broker) Subscribe(eventType EventType, subscriber EventSubscriber) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.subscribers[string(eventType)] = append(b.subscribers[string(eventType)], subscriber)
}

/* notifySubscribers notifies local subscribers */
func (b *Broker) notifySubscribers(ctx context.Context, event Event) {
	b.mu.RLock()
	subscribers := b.subscribers[event.Type]
	b.mu.RUnlock()

	for _, subscriber := range subscribers {
		if err := subscriber(ctx, event); err != nil {
			metrics.WarnWithContext(ctx, "Subscriber error", map[string]interface{}{
				"event_type": event.Type,
				"error":      err.Error(),
			})
		}
	}
}

/* storeEvent stores event in database for event sourcing */
func (b *Broker) storeEvent(ctx context.Context, event Event) error {
	query := `INSERT INTO neurondb_agent.events
		(id, type, timestamp, source, data, created_at)
		VALUES ($1, $2, $3, $4, $5::jsonb, NOW())`

	dataJSON, err := json.Marshal(event.Data)
	if err != nil {
		return fmt.Errorf("event storage failed: json_marshal_error=true, error=%w", err)
	}

	_, err = b.queries.DB.ExecContext(ctx, query,
		event.ID,
		event.Type,
		event.Timestamp,
		event.Source,
		dataJSON,
	)

	return err
}

/* PostgreSQLBackend implements EventBackend using PostgreSQL */
type PostgreSQLBackend struct {
	queries  *db.Queries
	listener *pq.Listener
	mu       sync.RWMutex
	handlers map[string]EventHandler
	stopChan chan struct{}
}

/* NewPostgreSQLBackend creates a new PostgreSQL backend */
func NewPostgreSQLBackend(queries *db.Queries) *PostgreSQLBackend {
	return &PostgreSQLBackend{
		queries:  queries,
		handlers: make(map[string]EventHandler),
		stopChan: make(chan struct{}),
	}
}

/* Publish publishes event via PostgreSQL NOTIFY */
func (pg *PostgreSQLBackend) Publish(ctx context.Context, topic string, event Event) error {
	query := `SELECT pg_notify($1, $2)`

	eventJSON, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("postgresql publish failed: json_marshal_error=true, error=%w", err)
	}

	_, err = pg.queries.DB.ExecContext(ctx, query, topic, string(eventJSON))
	return err
}

/* Subscribe subscribes to events via PostgreSQL LISTEN */
func (pg *PostgreSQLBackend) Subscribe(ctx context.Context, topic string, handler EventHandler) error {
	pg.mu.Lock()
	defer pg.mu.Unlock()

	/* Store handler for this topic */
	pg.handlers[topic] = handler

	/* Initialize listener if not already initialized */
	if pg.listener == nil {
		/* Get connection string from database */
		connStr := ""
		if pg.queries.DB != nil {
			var dbname, usr, host, port string
			err := pg.queries.DB.QueryRowContext(ctx, `
				SELECT 
					current_database() as database,
					current_user as user,
					inet_server_addr()::text as host,
					inet_server_port()::text as port
			`).Scan(&dbname, &usr, &host, &port)
			if err == nil {
				if host != "" && port != "" {
					connStr = fmt.Sprintf("host=%s port=%s dbname=%s user=%s sslmode=disable", host, port, dbname, usr)
				} else {
					connStr = fmt.Sprintf("dbname=%s user=%s sslmode=disable", dbname, usr)
				}
			} else {
				/* Fallback */
				if err := pg.queries.DB.QueryRowContext(ctx, "SELECT current_database(), current_user").Scan(&dbname, &usr); err == nil {
					connStr = fmt.Sprintf("dbname=%s user=%s sslmode=disable", dbname, usr)
				}
			}
		}

		if connStr == "" {
			return fmt.Errorf("postgresql subscribe failed: could_not_build_connection_string=true")
		}

		/* Create pq.Listener for LISTEN/NOTIFY */
		reportProblem := func(ev pq.ListenerEventType, err error) {
			if err != nil {
				metrics.WarnWithContext(ctx, "PostgreSQL LISTEN error", map[string]interface{}{
					"event": int(ev),
					"error": err.Error(),
				})
			}
		}

		listener := pq.NewListener(connStr, 10*time.Second, time.Minute, reportProblem)
		if err := listener.Listen(topic); err != nil {
			return fmt.Errorf("postgresql subscribe failed: listen_error=true, topic='%s', error=%w", topic, err)
		}

		pg.listener = listener

		/* Start listening goroutine */
		go pg.listenForNotifications(ctx)
	} else {
		/* Add additional topic to existing listener */
		if err := pg.listener.Listen(topic); err != nil {
			return fmt.Errorf("postgresql subscribe failed: listen_error=true, topic='%s', error=%w", topic, err)
		}
	}

	metrics.InfoWithContext(ctx, "PostgreSQL event subscription enabled", map[string]interface{}{
		"topic": topic,
	})

	return nil
}

/* listenForNotifications listens for PostgreSQL NOTIFY events */
func (pg *PostgreSQLBackend) listenForNotifications(ctx context.Context) {
	notificationChan := pg.listener.Notify
	for {
		select {
		case <-ctx.Done():
			return
		case <-pg.stopChan:
			return
		case notification := <-notificationChan:
			if notification != nil {
				/* Parse event from notification payload */
				var event Event
				if err := json.Unmarshal([]byte(notification.Extra), &event); err != nil {
					metrics.WarnWithContext(ctx, "Failed to parse event from notification", map[string]interface{}{
						"channel": notification.Channel,
						"error":   err.Error(),
					})
					continue
				}

				/* Get handler for this topic */
				pg.mu.RLock()
				handler, ok := pg.handlers[notification.Channel]
				pg.mu.RUnlock()

				if ok && handler != nil {
					/* Call handler */
					if err := handler(ctx, event); err != nil {
						metrics.WarnWithContext(ctx, "Event handler error", map[string]interface{}{
							"event_type": event.Type,
							"channel":    notification.Channel,
							"error":      err.Error(),
						})
					}
				}
			}
		}
	}
}

/* Close closes the backend */
func (pg *PostgreSQLBackend) Close() error {
	pg.mu.Lock()
	defer pg.mu.Unlock()

	if pg.listener != nil {
		close(pg.stopChan)
		pg.listener.Close()
		pg.listener = nil
	}

	pg.handlers = make(map[string]EventHandler)
	return nil
}

