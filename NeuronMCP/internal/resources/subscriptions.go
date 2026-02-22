/*-------------------------------------------------------------------------
 *
 * subscriptions.go
 *    Resource subscription manager for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/subscriptions.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/neurondb/NeuronMCP/internal/logging"
)

/* Subscription represents a resource subscription */
type Subscription struct {
	ID        string
	URI       string
	Filter    string /* Optional filter pattern (e.g., "schema:*", "model:*") */
	Callback  func(*ResourceUpdate)
	CreatedAt time.Time
}

/* ResourceUpdate represents a resource update */
type ResourceUpdate struct {
	URI       string      `json:"uri"`
	Type      string      `json:"type"` /* created, updated, deleted */
	Content   interface{} `json:"content,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
}

/* SubscriptionManager manages resource subscriptions */
type SubscriptionManager struct {
	subscriptions map[string][]*Subscription
	mu            sync.RWMutex
	wg            sync.WaitGroup  /* Track active goroutines for cleanup */
	shutdown      chan struct{}   /* Signal for shutdown */
	shutdownOnce  sync.Once       /* Ensure shutdown is called only once */
	logger        *logging.Logger /* Logger for error reporting */
}

/* NewSubscriptionManager creates a new subscription manager */
func NewSubscriptionManager() *SubscriptionManager {
	return &SubscriptionManager{
		subscriptions: make(map[string][]*Subscription),
		shutdown:      make(chan struct{}),
		logger:        nil, /* Logger can be set via SetLogger if needed */
	}
}

/* SetLogger sets the logger for the subscription manager */
func (m *SubscriptionManager) SetLogger(logger *logging.Logger) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.logger = logger
}

/* Subscribe subscribes to resource updates */
func (m *SubscriptionManager) Subscribe(uri string, callback func(*ResourceUpdate)) (string, error) {
	return m.SubscribeWithFilter(uri, "", callback)
}

/* SubscribeWithFilter subscribes to resource updates with an optional filter pattern */
/* Filter patterns support wildcards: "schema:*" matches all schema resources, "model:my-model" matches specific model */
func (m *SubscriptionManager) SubscribeWithFilter(uri string, filter string, callback func(*ResourceUpdate)) (string, error) {
	if uri == "" {
		return "", fmt.Errorf("URI cannot be empty")
	}
	if callback == nil {
		return "", fmt.Errorf("callback cannot be nil")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	subID := fmt.Sprintf("%s-%d", uri, time.Now().UnixNano())
	sub := &Subscription{
		ID:        subID,
		URI:       uri,
		Filter:    filter,
		Callback:  callback,
		CreatedAt: time.Now(),
	}

	m.subscriptions[uri] = append(m.subscriptions[uri], sub)
	return subID, nil
}

/* matchesFilter checks if a URI matches a filter pattern */
func matchesFilter(uri, filter string) bool {
	if filter == "" {
		return true /* No filter means match all */
	}

	/* Simple wildcard matching */
	if strings.Contains(filter, "*") {
		pattern := strings.ReplaceAll(filter, "*", ".*")
		matched, _ := regexp.MatchString("^"+pattern+"$", uri)
		return matched
	}

	/* Exact match */
	return uri == filter
}

/* Unsubscribe unsubscribes from resource updates */
func (m *SubscriptionManager) Unsubscribe(subID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for uri, subs := range m.subscriptions {
		for i, sub := range subs {
			if sub.ID == subID {
				m.subscriptions[uri] = append(subs[:i], subs[i+1:]...)
				if len(m.subscriptions[uri]) == 0 {
					delete(m.subscriptions, uri)
				}
				return nil
			}
		}
	}

	return fmt.Errorf("subscription not found: %s", subID)
}

/* Notify notifies subscribers of a resource update and invalidates cache if needed */
func (m *SubscriptionManager) Notify(uri string, updateType string, content interface{}) {
	if uri == "" {
		return /* URI is required */
	}
	if updateType == "" {
		updateType = "update" /* Default update type */
	}

	m.mu.RLock()
	subs := m.subscriptions[uri]
	m.mu.RUnlock()

	if len(subs) == 0 {
		return /* No subscribers for this URI */
	}

	update := &ResourceUpdate{
		URI:       uri,
		Type:      updateType,
		Content:   content,
		Timestamp: time.Now(),
	}

	/* Notify all subscribers in separate goroutines */
	for _, sub := range subs {
		if sub == nil || sub.Callback == nil {
			continue /* Skip invalid subscriptions */
		}

		/* Apply filter if specified */
		if sub.Filter != "" && !matchesFilter(uri, sub.Filter) {
			continue /* Skip if filter doesn't match */
		}

		/* Check if shutdown was requested */
		select {
		case <-m.shutdown:
			/* Shutdown requested, don't start new goroutines */
			return
		default:
		}

		/* Launch callback in goroutine to avoid blocking */
		/* Use a timeout context to prevent goroutine leak if callback hangs */
		m.wg.Add(1)
		go func(callback func(*ResourceUpdate), upd *ResourceUpdate, subID string) {
			defer m.wg.Done()
			defer func() {
				if r := recover(); r != nil {
					/* Recover from panics in callbacks and log them */
					if m.logger != nil {
						m.logger.Warn("Panic recovered in subscription callback", map[string]interface{}{
							"subscription_id": subID,
							"uri":             upd.URI,
							"panic_value":     fmt.Sprintf("%v", r),
						})
					}
				}
			}()

			/* Create a timeout context for the callback (30 seconds max) */
			const subscriptionCallbackTimeout = 30 * time.Second
			ctx, cancel := context.WithTimeout(context.Background(), subscriptionCallbackTimeout)
			defer cancel()

			/* Execute callback with timeout protection */
			/* Use a single goroutine with proper context cancellation */
			done := make(chan struct{}, 1)
			errChan := make(chan error, 1)

			go func() {
				defer func() {
					if r := recover(); r != nil {
						/* Recover from panics in callback execution and log them */
						if m.logger != nil {
							m.logger.Warn("Panic recovered during callback execution", map[string]interface{}{
								"subscription_id": subID,
								"uri":             upd.URI,
								"panic_value":     fmt.Sprintf("%v", r),
							})
						}
						errChan <- fmt.Errorf("panic in callback: %v", r)
					}
				}()

				/* Check if context is already cancelled before executing */
				select {
				case <-ctx.Done():
					return
				case <-m.shutdown:
					return
				default:
				}

				callback(upd)
				done <- struct{}{}
			}()

			select {
			case <-done:
				/* Callback completed successfully */
			case err := <-errChan:
				/* Panic occurred in callback */
				if m.logger != nil {
					m.logger.Warn("Callback execution error", map[string]interface{}{
						"subscription_id": subID,
						"uri":             upd.URI,
						"error":           err.Error(),
					})
				}
			case <-ctx.Done():
				/* Callback timed out */
				if m.logger != nil {
					m.logger.Warn("Subscription callback timeout", map[string]interface{}{
						"subscription_id": subID,
						"uri":             upd.URI,
						"timeout_seconds": int(subscriptionCallbackTimeout.Seconds()),
					})
				}
			case <-m.shutdown:
				/* Shutdown requested, exit immediately */
				cancel()
			}
		}(sub.Callback, update, sub.ID)
	}
}

/* ListSubscriptions lists all subscriptions */
func (m *SubscriptionManager) ListSubscriptions() []*Subscription {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var allSubs []*Subscription
	for _, subs := range m.subscriptions {
		allSubs = append(allSubs, subs...)
	}

	return allSubs
}

/* Shutdown gracefully shuts down the subscription manager */
/* Waits for all active goroutines to complete or timeout */
func (m *SubscriptionManager) Shutdown(timeout time.Duration) error {
	var shutdownErr error
	m.shutdownOnce.Do(func() {
		/* Signal shutdown */
		close(m.shutdown)

		/* Wait for all goroutines to complete with timeout */
		done := make(chan struct{})
		go func() {
			m.wg.Wait()
			close(done)
		}()

		select {
		case <-done:
			/* All goroutines completed */
		case <-time.After(timeout):
			/* Timeout waiting for goroutines */
			shutdownErr = fmt.Errorf("subscription manager shutdown timeout after %v", timeout)
		}
	})
	return shutdownErr
}
