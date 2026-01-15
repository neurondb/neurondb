package middleware

import (
	"crypto/rand"
	"encoding/base64"
	"net/http"
	"time"
)

/* CSRF token storage (in-memory, should use Redis in production) */
var csrfTokens = make(map[string]time.Time)
var csrfTokenExpiry = 24 * time.Hour

/* GenerateCSRFToken generates a new CSRF token */
func GenerateCSRFToken() (string, error) {
	token := make([]byte, 32)
	if _, err := rand.Read(token); err != nil {
		return "", err
	}
	tokenStr := base64.URLEncoding.EncodeToString(token)
	csrfTokens[tokenStr] = time.Now().Add(csrfTokenExpiry)
	return tokenStr, nil
}

/* ValidateCSRFToken validates a CSRF token */
func ValidateCSRFToken(token string) bool {
	expiry, exists := csrfTokens[token]
	if !exists {
		return false
	}
	if time.Now().After(expiry) {
		delete(csrfTokens, token)
		return false
	}
	return true
}

/* CSRFMiddleware provides CSRF protection for state-changing operations */
func CSRFMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			/* Only check CSRF for state-changing methods */
			if r.Method == "GET" || r.Method == "HEAD" || r.Method == "OPTIONS" {
				next.ServeHTTP(w, r)
				return
			}

			/* Get token from header or form */
			token := r.Header.Get("X-CSRF-Token")
			if token == "" {
				token = r.FormValue("csrf_token")
			}

			/* Validate token */
			if !ValidateCSRFToken(token) {
				http.Error(w, "Invalid or missing CSRF token", http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

/* CleanupExpiredTokens removes expired CSRF tokens (should be called periodically) */
func CleanupExpiredTokens() {
	now := time.Now()
	for token, expiry := range csrfTokens {
		if now.After(expiry) {
			delete(csrfTokens, token)
		}
	}
}








