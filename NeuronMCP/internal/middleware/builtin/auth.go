/*-------------------------------------------------------------------------
 *
 * auth.go
 *    Authentication middleware for NeuronMCP
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/middleware/builtin/auth.go
 *
 *-------------------------------------------------------------------------
 */

package builtin

import (
	"context"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/neurondb/NeuronMCP/internal/context/contextkeys"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* AuthConfig holds authentication configuration */
type AuthConfig struct {
	Enabled       bool
	APIKeyHashes  map[string]string /* SHA256 hex hash of API key -> user mapping */
	JWTSecret     string
	JWTPublicKey  *rsa.PublicKey
	OAuth2Config  *OAuth2Config
}

/* HashAPIKey returns SHA256 hex hash of an API key for secure storage */
func HashAPIKey(apiKey string) string {
	h := sha256.Sum256([]byte(apiKey))
	return hex.EncodeToString(h[:])
}

/* SetAPIKeysFromPlaintext hashes and stores API keys from plaintext format.
 * This is a migration helper function. In production, API keys should be
 * provided already hashed via APIKeyHashes map.
 */
func (c *AuthConfig) SetAPIKeysFromPlaintext(plain map[string]string) {
	if c.APIKeyHashes == nil {
		c.APIKeyHashes = make(map[string]string)
	}
	for k, u := range plain {
		c.APIKeyHashes[HashAPIKey(k)] = u
	}
}

/* OAuth2Config holds OAuth2 configuration */
type OAuth2Config struct {
	ClientID     string
	ClientSecret string
	Issuer       string
	Audience     string
}

/* AuthMiddleware provides authentication */
type AuthMiddleware struct {
	config *AuthConfig
	logger *logging.Logger
}

/* NewAuthMiddleware creates a new authentication middleware */
func NewAuthMiddleware(config *AuthConfig, logger *logging.Logger) middleware.Middleware {
	return &AuthMiddleware{
		config: config,
		logger: logger,
	}
}

/* Name returns the middleware name */
func (m *AuthMiddleware) Name() string {
	return "auth"
}

/* Order returns the middleware order */
func (m *AuthMiddleware) Order() int {
	return 0
}

/* Enabled returns whether the middleware is enabled */
func (m *AuthMiddleware) Enabled() bool {
	return m.config.Enabled
}

/* Execute handles authentication */
func (m *AuthMiddleware) Execute(ctx context.Context, req *middleware.MCPRequest, next middleware.Handler) (*middleware.MCPResponse, error) {
	if !m.config.Enabled {
		return next(ctx, req)
	}

	/* Extract token from request */
	token := m.extractToken(req)
	if token == "" {
		return &middleware.MCPResponse{
			Content: []middleware.ContentBlock{
				{Type: "text", Text: "Authentication required"},
			},
			IsError: true,
		}, nil
	}

	/* Try API key authentication (hashed storage only) */
	if m.config.APIKeyHashes != nil {
		if user, ok := m.config.APIKeyHashes[HashAPIKey(token)]; ok {
			ctx = context.WithValue(ctx, contextkeys.UserKey{}, user)
			return next(ctx, req)
		}
	}

	/* Try JWT authentication */
	if m.config.JWTSecret != "" || m.config.JWTPublicKey != nil {
		user, err := m.validateJWT(token)
		if err == nil {
			ctx = context.WithValue(ctx, contextkeys.UserKey{}, user)
			return next(ctx, req)
		}
		m.logger.Debug("JWT validation failed", map[string]interface{}{
			"error": err.Error(),
		})
	}

	/* Authentication failed */
	return &middleware.MCPResponse{
		Content: []middleware.ContentBlock{
			{Type: "text", Text: "Invalid authentication token"},
		},
		IsError: true,
	}, nil
}

/* extractToken extracts token from request with strict precedence; rejects multiple sources */
/* Precedence: Authorization Bearer > X-API-Key > metadata apiKey/token > params token/apiKey */
func (m *AuthMiddleware) extractToken(req *middleware.MCPRequest) string {
	type source struct {
		token string
		name  string
	}
	var candidates []source

	add := func(t, name string) {
		if t != "" {
			candidates = append(candidates, source{t, name})
		}
	}

	/* 1. Authorization Bearer */
	if req.Metadata != nil {
		for _, k := range []string{"authorization", "Authorization"} {
			if auth, ok := req.Metadata[k].(string); ok {
				add(m.extractBearerToken(auth), "Authorization")
				break
			}
		}
	}

	/* 2. X-API-Key (metadata from headers) */
	if req.Metadata != nil {
		for _, k := range []string{"X-Api-Key", "x-api-key", "apiKey"} {
			if v, ok := req.Metadata[k].(string); ok && v != "" {
				add(v, "X-API-Key")
				break
			}
		}
	}

	/* 3. metadata token */
	if req.Metadata != nil {
		if v, ok := req.Metadata["token"].(string); ok && v != "" {
			add(v, "metadata.token")
		}
	}

	/* 4. params token / apiKey */
	if req.Params != nil {
		if v, ok := req.Params["token"].(string); ok && v != "" {
			add(v, "params.token")
		} else if v, ok := req.Params["apiKey"].(string); ok && v != "" {
			add(v, "params.apiKey")
		}
	}

	if len(candidates) == 0 {
		return ""
	}
	/* Reject if multiple token sources present (potential bypass) */
	if len(candidates) > 1 {
		names := make([]string, len(candidates))
		for i := range candidates {
			names[i] = candidates[i].name
		}
		m.logger.Warn("Multiple token sources provided; rejecting", map[string]interface{}{
			"sources": names,
		})
		return ""
	}
	return candidates[0].token
}

/* extractBearerToken extracts bearer token from Authorization header */
func (m *AuthMiddleware) extractBearerToken(auth string) string {
	if strings.HasPrefix(auth, "Bearer ") {
		return strings.TrimPrefix(auth, "Bearer ")
	}
	return ""
}

/* validateJWT validates a JWT token */
func (m *AuthMiddleware) validateJWT(tokenString string) (string, error) {
	var key interface{}
	if m.config.JWTPublicKey != nil {
		key = m.config.JWTPublicKey
	} else if m.config.JWTSecret != "" {
		key = []byte(m.config.JWTSecret)
	} else {
		return "", fmt.Errorf("no JWT key configured")
	}

	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		/* Check signing method */
		method := token.Method
		if method == jwt.SigningMethodRS256 || method == jwt.SigningMethodRS384 || method == jwt.SigningMethodRS512 ||
			method == jwt.SigningMethodPS256 || method == jwt.SigningMethodPS384 || method == jwt.SigningMethodPS512 {
			if m.config.JWTPublicKey != nil {
				return m.config.JWTPublicKey, nil
			}
			return nil, fmt.Errorf("RSA public key not configured for signing method: %s", method.Alg())
		}
		if method == jwt.SigningMethodHS256 || method == jwt.SigningMethodHS384 || method == jwt.SigningMethodHS512 {
			return key, nil
		}
		return nil, fmt.Errorf("unexpected signing method: %s", method.Alg())
	})

	if err != nil {
		return "", err
	}

	if !token.Valid {
		return "", fmt.Errorf("invalid token")
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return "", fmt.Errorf("invalid claims")
	}

	/* Check expiration with clock skew tolerance (5 minutes) */
	if exp, ok := claims["exp"].(float64); ok {
		clockSkew := 5 * time.Minute
		expTime := time.Unix(int64(exp), 0)
		if time.Now().Add(clockSkew).After(expTime) {
			return "", fmt.Errorf("token expired")
		}
	}

	/* Extract user/subject */
	if sub, ok := claims["sub"].(string); ok {
		return sub, nil
	}
	if user, ok := claims["user"].(string); ok {
		return user, nil
	}

	return "", fmt.Errorf("no user in token")
}

/* LoadRSAPublicKey loads an RSA public key from PEM or base64 */
func LoadRSAPublicKey(keyData string) (*rsa.PublicKey, error) {
	/* Try PEM format first */
	block, _ := pem.Decode([]byte(keyData))
	if block != nil {
		pub, err := x509.ParsePKIXPublicKey(block.Bytes)
		if err != nil {
			return nil, err
		}
		if rsaPub, ok := pub.(*rsa.PublicKey); ok {
			return rsaPub, nil
		}
		return nil, fmt.Errorf("not an RSA public key")
	}

	/* Try base64 */
	decoded, err := base64.StdEncoding.DecodeString(keyData)
	if err == nil {
		pub, err := x509.ParsePKIXPublicKey(decoded)
		if err != nil {
			return nil, err
		}
		if rsaPub, ok := pub.(*rsa.PublicKey); ok {
			return rsaPub, nil
		}
		return nil, fmt.Errorf("not an RSA public key")
	}

	return nil, fmt.Errorf("invalid key format")
}

