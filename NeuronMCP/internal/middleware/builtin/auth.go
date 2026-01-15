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
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/neurondb/NeuronMCP/internal/logging"
	"github.com/neurondb/NeuronMCP/internal/middleware"
)

/* AuthConfig holds authentication configuration */
type AuthConfig struct {
	Enabled      bool
	APIKeys      map[string]string  /* API key -> user mapping */
	JWTSecret    string
	JWTPublicKey *rsa.PublicKey
	OAuth2Config *OAuth2Config
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

	/* Try API key authentication first */
	if m.config.APIKeys != nil {
		if user, ok := m.config.APIKeys[token]; ok {
			/* Add user to context */
			ctx = context.WithValue(ctx, "user", user)
			return next(ctx, req)
		}
	}

	/* Try JWT authentication */
	if m.config.JWTSecret != "" || m.config.JWTPublicKey != nil {
		user, err := m.validateJWT(token)
		if err == nil {
			ctx = context.WithValue(ctx, "user", user)
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

/* extractToken extracts token from request */
/* Supports both stdio (via metadata/params) and HTTP (via headers extracted to metadata) */
func (m *AuthMiddleware) extractToken(req *middleware.MCPRequest) string {
	/* Check metadata (populated from HTTP headers by HTTP transport, or from stdio metadata) */
	if req.Metadata != nil {
		/* Check for API key in metadata (from X-API-Key header or apiKey metadata) */
		if apiKey, ok := req.Metadata["apiKey"].(string); ok {
			return apiKey
		}
		/* Check for X-Api-Key header (case-insensitive header extraction) */
		if apiKey, ok := req.Metadata["X-Api-Key"].(string); ok {
			return apiKey
		}
		if apiKey, ok := req.Metadata["x-api-key"].(string); ok {
			return apiKey
		}
		/* Check for token in metadata */
		if token, ok := req.Metadata["token"].(string); ok {
			return token
		}
		/* Check for Authorization header (Bearer token) */
		if auth, ok := req.Metadata["authorization"].(string); ok {
			return m.extractBearerToken(auth)
		}
		if auth, ok := req.Metadata["Authorization"].(string); ok {
			return m.extractBearerToken(auth)
		}
	}

	/* Check params */
	if req.Params != nil {
		if token, ok := req.Params["token"].(string); ok {
			return token
		}
		if apiKey, ok := req.Params["apiKey"].(string); ok {
			return apiKey
		}
	}

	return ""
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

	/* Check expiration */
	if exp, ok := claims["exp"].(float64); ok {
		if time.Now().Unix() > int64(exp) {
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

