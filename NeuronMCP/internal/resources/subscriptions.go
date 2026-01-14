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
	"sync"
	"time"
)

/* Subscription represents a resource subscription */
type Subscription struct {
	ID        string
	URI       string
	Callback  func(*ResourceUpdate)
	CreatedAt time.Time
}

/* ResourceUpdate represents a resource update */
type ResourceUpdate struct {
	URI      string                 `json:"uri"`
	Type     string                 `json:"type"` /* created, updated, deleted */
	Content  interface{}            `json:"content,omitempty"`
	Timestamp time.Time             `json:"timestamp"`
}

/* SubscriptionManager manages resource subscriptions */
type SubscriptionManager struct {
	subscriptions map[string][]*Subscription
	mu            sync.RWMutex
}

/* NewSubscriptionManager creates a new subscription manager */
func NewSubscriptionManager() *SubscriptionManager {
	return &SubscriptionManager{
		subscriptions: make(map[string][]*Subscription),
	}
}

/* Subscribe subscribes to resource updates */
func (m *SubscriptionManager) Subscribe(uri string, callback func(*ResourceUpdate)) (string, error) {
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
		Callback:  callback,
		CreatedAt: time.Now(),
	}

	m.subscriptions[uri] = append(m.subscriptions[uri], sub)
	return subID, nil
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

/* Notify notifies subscribers of a resource update */
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
		/* Launch callback in goroutine to avoid blocking */
		/* Use a timeout context to prevent goroutine leak if callback hangs */
		go func(callback func(*ResourceUpdate), upd *ResourceUpdate) {
			defer func() {
				if r := recover(); r != nil {
					/* Recover from panics in callbacks */
				}
			}()
			
			/* Create a timeout context for the callback (30 seconds max) */
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			
			/* Execute callback in a goroutine that respects timeout */
			done := make(chan struct{}, 1)
			go func() {
				callback(upd)
				done <- struct{}{}
			}()
			
			select {
			case <-done:
				/* Callback completed successfully */
			case <-ctx.Done():
				/* Callback timed out, goroutine will exit */
			}
		}(sub.Callback, update)
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

