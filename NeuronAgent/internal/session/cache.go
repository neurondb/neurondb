package session

import (
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/neurondb/NeuronAgent/internal/db"
)

type Cache struct {
	sessions    map[uuid.UUID]*cachedSession
	mu          sync.RWMutex
	ttl         time.Duration
	cleanupTicker *time.Ticker
	stop        chan struct{}
}

type cachedSession struct {
	session   *db.Session
	expiresAt time.Time
}

func NewCache(ttl time.Duration) *Cache {
	cache := &Cache{
		sessions:      make(map[uuid.UUID]*cachedSession),
		ttl:           ttl,
		cleanupTicker: time.NewTicker(ttl / 2),
		stop:          make(chan struct{}),
	}

	// Start cleanup goroutine
	go cache.runCleanup()

	return cache
}

func (c *Cache) Set(id uuid.UUID, session *db.Session) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.sessions[id] = &cachedSession{
		session:   session,
		expiresAt: time.Now().Add(c.ttl),
	}
}

func (c *Cache) Get(id uuid.UUID) *db.Session {
	c.mu.RLock()
	defer c.mu.RUnlock()

	cached, exists := c.sessions[id]
	if !exists {
		return nil
	}

	if time.Now().After(cached.expiresAt) {
		// Expired, remove it
		c.mu.RUnlock()
		c.mu.Lock()
		delete(c.sessions, id)
		c.mu.Unlock()
		c.mu.RLock()
		return nil
	}

	return cached.session
}

func (c *Cache) Delete(id uuid.UUID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.sessions, id)
}

func (c *Cache) runCleanup() {
	for {
		select {
		case <-c.stop:
			return
		case <-c.cleanupTicker.C:
			c.mu.Lock()
			now := time.Now()
			for id, cached := range c.sessions {
				if now.After(cached.expiresAt) {
					delete(c.sessions, id)
				}
			}
			c.mu.Unlock()
		}
	}
}

func (c *Cache) Close() {
	c.cleanupTicker.Stop()
	close(c.stop)
}

