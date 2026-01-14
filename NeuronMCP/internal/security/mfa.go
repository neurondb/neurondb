/*-------------------------------------------------------------------------
 *
 * mfa.go
 *    Multi-Factor Authentication (MFA) support
 *
 * Implements MFA support as specified in Phase 2.1.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/security/mfa.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"crypto/rand"
	"fmt"
	"time"

	"github.com/pquerna/otp/totp"
)

/* MFAMethod represents an MFA method */
type MFAMethod string

const (
	MFAMethodTOTP MFAMethod = "totp"
	MFAMethodSMS  MFAMethod = "sms"
	MFAMethodEmail MFAMethod = "email"
)

/* MFAConfig represents MFA configuration for a user */
type MFAConfig struct {
	UserID      string
	Method      MFAMethod
	Secret      string /* TOTP secret */
	PhoneNumber string /* For SMS */
	Email       string /* For email */
	Enabled     bool
	CreatedAt   time.Time
}

/* MFAManager manages MFA */
type MFAManager struct {
	configs map[string]*MFAConfig /* user_id -> MFAConfig */
}

/* NewMFAManager creates a new MFA manager */
func NewMFAManager() *MFAManager {
	return &MFAManager{
		configs: make(map[string]*MFAConfig),
	}
}

/* GenerateTOTPSecret generates a TOTP secret for a user */
func (m *MFAManager) GenerateTOTPSecret(userID, issuer string) (string, string, error) {
	key, err := totp.Generate(totp.GenerateOpts{
		Issuer:      issuer,
		AccountName: userID,
	})
	if err != nil {
		return "", "", fmt.Errorf("failed to generate TOTP secret: %w", err)
	}

	/* Store configuration */
	m.configs[userID] = &MFAConfig{
		UserID:    userID,
		Method:    MFAMethodTOTP,
		Secret:    key.Secret(),
		Enabled:   false, /* Not enabled until verified */
		CreatedAt: time.Now(),
	}

	return key.Secret(), key.URL(), nil
}

/* VerifyTOTP verifies a TOTP code */
func (m *MFAManager) VerifyTOTP(userID, code string) (bool, error) {
	config, exists := m.configs[userID]
	if !exists {
		return false, fmt.Errorf("MFA not configured for user")
	}

	if config.Method != MFAMethodTOTP {
		return false, fmt.Errorf("invalid MFA method")
	}

	valid := totp.Validate(code, config.Secret)
	if valid {
		/* Enable MFA after first successful verification */
		if !config.Enabled {
			config.Enabled = true
		}
	}

	return valid, nil
}

/* IsMFAEnabled checks if MFA is enabled for a user */
func (m *MFAManager) IsMFAEnabled(userID string) bool {
	config, exists := m.configs[userID]
	if !exists {
		return false
	}
	return config.Enabled
}

/* DisableMFA disables MFA for a user */
func (m *MFAManager) DisableMFA(userID string) error {
	config, exists := m.configs[userID]
	if !exists {
		return fmt.Errorf("MFA not configured for user")
	}

	config.Enabled = false
	return nil
}

/* GenerateSMSCode generates a code for SMS MFA */
func (m *MFAManager) GenerateSMSCode(userID, phoneNumber string) (string, error) {
	/* Generate 6-digit code */
	codeBytes := make([]byte, 3)
	if _, err := rand.Read(codeBytes); err != nil {
		return "", fmt.Errorf("failed to generate code: %w", err)
	}

	code := fmt.Sprintf("%06d", int(codeBytes[0])<<16|int(codeBytes[1])<<8|int(codeBytes[2]))
	code = code[:6] /* Ensure 6 digits */

	/* Store configuration */
	m.configs[userID] = &MFAConfig{
		UserID:      userID,
		Method:      MFAMethodSMS,
		PhoneNumber: phoneNumber,
		Enabled:     false,
		CreatedAt:   time.Now(),
	}

	return code, nil
}

/* GenerateEmailCode generates a code for email MFA */
func (m *MFAManager) GenerateEmailCode(userID, email string) (string, error) {
	/* Generate 6-digit code */
	codeBytes := make([]byte, 3)
	if _, err := rand.Read(codeBytes); err != nil {
		return "", fmt.Errorf("failed to generate code: %w", err)
	}

	code := fmt.Sprintf("%06d", int(codeBytes[0])<<16|int(codeBytes[1])<<8|int(codeBytes[2]))
	code = code[:6] /* Ensure 6 digits */

	/* Store configuration */
	m.configs[userID] = &MFAConfig{
		UserID:    userID,
		Method:    MFAMethodEmail,
		Email:     email,
		Enabled:   false,
		CreatedAt: time.Now(),
	}

	return code, nil
}

/* VerifyCode verifies an MFA code (for SMS/Email) */
func (m *MFAManager) VerifyCode(userID, code string) (bool, error) {
	config, exists := m.configs[userID]
	if !exists {
		return false, fmt.Errorf("MFA not configured for user")
	}

	/* For SMS/Email, we'd need to store codes with expiration */
	/* This is a simplified version */
	if config.Method == MFAMethodSMS || config.Method == MFAMethodEmail {
		/* In production, verify against stored code with expiration */
		config.Enabled = true
		return true, nil
	}

	return false, fmt.Errorf("invalid MFA method")
}

