/*-------------------------------------------------------------------------
 *
 * oidc.go
 *    OIDC device flow authentication for NeuronMCP
 *
 * Implements OIDC device authorization grant for secure laptop client authentication.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/auth/oidc.go
 *
 *-------------------------------------------------------------------------
 */

package auth

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"sync"
	"time"
)

/* DeviceCode represents a device authorization code */
type DeviceCode struct {
	DeviceCode      string
	UserCode        string
	VerificationURI string
	ExpiresIn       int
	Interval        int
	CreatedAt       time.Time
	UserID          string
	Scopes          []string
	Verified        bool
}

/* OIDCProvider provides OIDC device flow authentication */
type OIDCProvider struct {
	deviceCodes map[string]*DeviceCode
	userCodes   map[string]*DeviceCode
	mu          sync.RWMutex
	tokenStore  map[string]*TokenInfo
	issuer      string
	clientID    string
}

/* TokenInfo represents a token issued via device flow */
type TokenInfo struct {
	AccessToken  string
	RefreshToken string
	ExpiresAt    time.Time
	UserID       string
	Scopes       []string
	OrgID        string
	ProjectID    string
}

/* NewOIDCProvider creates a new OIDC provider */
func NewOIDCProvider(issuer, clientID string) *OIDCProvider {
	return &OIDCProvider{
		deviceCodes: make(map[string]*DeviceCode),
		userCodes:   make(map[string]*DeviceCode),
		tokenStore:  make(map[string]*TokenInfo),
		issuer:      issuer,
		clientID:    clientID,
	}
}

/* InitiateDeviceFlow initiates the device authorization flow */
func (p *OIDCProvider) InitiateDeviceFlow(scopes []string) (*DeviceCode, error) {
	deviceCode, err := generateRandomCode(32)
	if err != nil {
		return nil, fmt.Errorf("failed to generate device code: %w", err)
	}

	userCode, err := generateUserCode()
	if err != nil {
		return nil, fmt.Errorf("failed to generate user code: %w", err)
	}

	code := &DeviceCode{
		DeviceCode:      deviceCode,
		UserCode:        userCode,
		VerificationURI: fmt.Sprintf("%s/device", p.issuer),
		ExpiresIn:       600, // 10 minutes
		Interval:        5,   // Poll every 5 seconds
		CreatedAt:       time.Now(),
		Scopes:          scopes,
		Verified:        false,
	}

	p.mu.Lock()
	p.deviceCodes[deviceCode] = code
	p.userCodes[userCode] = code
	p.mu.Unlock()

	return code, nil
}

/* VerifyDeviceCode verifies a user code and associates it with a user */
func (p *OIDCProvider) VerifyDeviceCode(userCode, userID string, orgID, projectID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	code, exists := p.userCodes[userCode]
	if !exists {
		return fmt.Errorf("invalid user code")
	}

	if time.Since(code.CreatedAt) > time.Duration(code.ExpiresIn)*time.Second {
		delete(p.deviceCodes, code.DeviceCode)
		delete(p.userCodes, userCode)
		return fmt.Errorf("device code expired")
	}

	code.Verified = true
	code.UserID = userID

	return nil
}

/* PollDeviceFlow polls for device authorization completion */
func (p *OIDCProvider) PollDeviceFlow(ctx context.Context, deviceCode string) (*TokenInfo, error) {
	p.mu.RLock()
	code, exists := p.deviceCodes[deviceCode]
	p.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("invalid device code")
	}

	if time.Since(code.CreatedAt) > time.Duration(code.ExpiresIn)*time.Second {
		p.mu.Lock()
		delete(p.deviceCodes, deviceCode)
		delete(p.userCodes, code.UserCode)
		p.mu.Unlock()
		return nil, fmt.Errorf("device code expired")
	}

	/* Check if verified */
	if !code.Verified {
		return nil, nil // Still pending
	}

	/* Generate access token */
	accessToken, err := generateRandomCode(32)
	if err != nil {
		return nil, fmt.Errorf("failed to generate access token: %w", err)
	}

	refreshToken, err := generateRandomCode(32)
	if err != nil {
		return nil, fmt.Errorf("failed to generate refresh token: %w", err)
	}

	tokenInfo := &TokenInfo{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresAt:    time.Now().Add(1 * time.Hour), // Short-lived: 1 hour
		UserID:       code.UserID,
		Scopes:       code.Scopes,
	}

	p.mu.Lock()
	p.tokenStore[accessToken] = tokenInfo
	p.mu.Unlock()

	/* Cleanup device codes */
	delete(p.deviceCodes, deviceCode)
	delete(p.userCodes, code.UserCode)

	return tokenInfo, nil
}

/* ValidateToken validates an access token */
func (p *OIDCProvider) ValidateToken(accessToken string) (*TokenInfo, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	tokenInfo, exists := p.tokenStore[accessToken]
	if !exists {
		return nil, fmt.Errorf("invalid access token")
	}

	if time.Now().After(tokenInfo.ExpiresAt) {
		return nil, fmt.Errorf("access token expired")
	}

	return tokenInfo, nil
}

/* RefreshToken refreshes an access token */
func (p *OIDCProvider) RefreshToken(refreshToken string) (*TokenInfo, error) {
	/* In a full implementation, this would look up the refresh token and issue a new access token */
	/* For now, return an error indicating refresh is not fully implemented */
	return nil, fmt.Errorf("token refresh not fully implemented - please re-authenticate")
}

/* generateRandomCode generates a random code */
func generateRandomCode(length int) (string, error) {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(bytes), nil
}

/* generateUserCode generates a user-friendly code (e.g., ABCD-EFGH) */
func generateUserCode() (string, error) {
	bytes := make([]byte, 4)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}

	code := ""
	for i, b := range bytes {
		if i == 2 {
			code += "-"
		}
		code += string([]byte{'A' + (b % 26)})
	}
	return code, nil
}












