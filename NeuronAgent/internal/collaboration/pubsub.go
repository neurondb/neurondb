/*-------------------------------------------------------------------------
 *
 * pubsub.go
 *    Publish-subscribe system for real-time updates
 *
 * Provides message broadcasting to workspace participants for real-time
 * collaboration updates.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/collaboration/pubsub.go
 *
 *-------------------------------------------------------------------------
 */

package collaboration

import (
	"context"
	"sync"
)

/* PubSub provides publish-subscribe functionality */
type PubSub struct {
	subscribers map[string]map[chan interface{}]bool
	mu          sync.RWMutex
}

/* NewPubSub creates a new pubsub instance */
func NewPubSub() *PubSub {
	return &PubSub{
		subscribers: make(map[string]map[chan interface{}]bool),
	}
}

/* Subscribe subscribes to a channel */
func (p *PubSub) Subscribe(channel string) chan interface{} {
	p.mu.Lock()
	defer p.mu.Unlock()

	ch := make(chan interface{}, 100)
	if p.subscribers[channel] == nil {
		p.subscribers[channel] = make(map[chan interface{}]bool)
	}
	p.subscribers[channel][ch] = true

	return ch
}

/* Unsubscribe unsubscribes from a channel */
func (p *PubSub) Unsubscribe(channel string, ch chan interface{}) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.subscribers[channel] != nil {
		delete(p.subscribers[channel], ch)
		close(ch)
	}
}

/* Publish publishes a message to a channel */
func (p *PubSub) Publish(ctx context.Context, channel string, message interface{}) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if p.subscribers[channel] != nil {
		for ch := range p.subscribers[channel] {
			select {
			case ch <- message:
			case <-ctx.Done():
				return
			default:
				/* Skip if channel is full */
			}
		}
	}
}
