# OIDC and Session Security Hardening

## Overview

This document describes the security hardening implemented for OIDC authentication and session management in NeuronDesktop.

## Security Improvements

### 1. Database-Backed Login Attempts

**Before**: Login attempts were stored in-memory (`map[string]*oidc.LoginAttempt`), which:
- Could not scale across multiple server instances
- Lost state on server restart
- Was vulnerable to race conditions

**After**: Login attempts are stored in the `login_attempts` table:
- Persists across server restarts
- Works with multiple server instances
- Includes expiration timestamps for automatic cleanup
- One-time use (deleted after successful callback)

**Implementation**:
- Migration: `009_oidc_hardening.sql` adds `redirect_uri` column
- Handler updates: `oidc.go` uses database queries instead of in-memory map
- Cleanup function: `cleanup_expired_login_attempts()` removes old attempts

### 2. Secure Cookie Configuration

**Cookie Security Settings**:
- `HttpOnly`: `true` - Prevents JavaScript access to cookies
- `Secure`: Configurable (default: `true` in production) - Only sent over HTTPS
- `SameSite`: Configurable (default: `Lax`) - CSRF protection

**Configuration**:
```go
sessionMgr := session.NewManager(
    database,
    accessTTL,        // Access token TTL (default: 15 minutes)
    refreshTTL,       // Refresh token TTL (default: 7 days)
    cookieDomain,     // Cookie domain
    cookieSecure,     // Secure flag (true for HTTPS)
    cookieSameSite,   // SameSite mode (Strict, Lax, None)
)
```

**Environment Variables**:
- `SESSION_COOKIE_SECURE`: Set to `true` in production
- `SESSION_COOKIE_SAME_SITE`: `Strict`, `Lax`, or `None` (default: `Lax`)

### 3. Token Rotation

**Refresh Token Rotation**:
- Each refresh token can only be used once
- New refresh token is generated on each refresh
- Old refresh token is revoked immediately
- Token reuse detection revokes entire session

**Implementation**:
```go
func (m *Manager) RefreshSession(ctx context.Context, refreshTokenString string) (*Session, string, string, error) {
    // 1. Validate refresh token
    // 2. Revoke old refresh token
    // 3. Generate new refresh token (with rotated_from reference)
    // 4. Return new access and refresh tokens
}
```

### 4. Session Validation

**Session Security**:
- Sessions are validated on every request
- Revoked sessions are immediately rejected
- Last seen timestamp is updated on each request
- Sessions can be revoked by admin or user

**Session Lifecycle**:
1. **Creation**: On successful OIDC login
2. **Validation**: On each authenticated request
3. **Refresh**: When access token expires
4. **Revocation**: On logout or security event

### 5. PKCE (Proof Key for Code Exchange)

**OIDC Flow Security**:
- Code verifier generated on client
- Code challenge sent to authorization server
- Code verifier validated on callback
- Prevents authorization code interception attacks

**Implementation**:
- Code verifier: 32 random bytes, base64 URL encoded
- Code challenge: SHA-256 hash of verifier
- Stored in `login_attempts` table with state/nonce

### 6. Audit Logging

**Security Events Logged**:
- Login attempts (success and failure)
- Token refresh events
- Session revocation
- OIDC callback processing

**Audit Log Schema**:
```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY,
    event_type TEXT NOT NULL,
    user_id UUID,
    ip_hash TEXT,
    user_agent_hash TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ
);
```

## Security Best Practices

### Production Configuration

1. **Enable HTTPS**:
   ```bash
   SESSION_COOKIE_SECURE=true
   ```

2. **Use Strict SameSite**:
   ```bash
   SESSION_COOKIE_SAME_SITE=Strict
   ```

3. **Set Appropriate TTLs**:
   ```bash
   SESSION_ACCESS_TTL=15m    # Short-lived access tokens
   SESSION_REFRESH_TTL=7d    # Longer-lived refresh tokens
   ```

4. **Enable Audit Logging**:
   - All authentication events are logged
   - Review audit logs regularly
   - Set up alerts for suspicious activity

5. **Cleanup Expired Attempts**:
   ```sql
   -- Run periodically (e.g., via cron)
   SELECT cleanup_expired_login_attempts();
   ```

### Security Checklist

- [ ] HTTPS enabled in production
- [ ] `SESSION_COOKIE_SECURE=true` in production
- [ ] `SESSION_COOKIE_SAME_SITE=Strict` or `Lax`
- [ ] Access token TTL ≤ 15 minutes
- [ ] Refresh token TTL ≤ 7 days
- [ ] Audit logging enabled
- [ ] Expired login attempts cleanup scheduled
- [ ] Session revocation on security events
- [ ] PKCE enabled for OIDC flow
- [ ] Database-backed login attempts

## Threat Mitigation

### CSRF (Cross-Site Request Forgery)
- **Mitigation**: SameSite cookie attribute
- **Implementation**: `SameSite=Lax` or `Strict`

### XSS (Cross-Site Scripting)
- **Mitigation**: HttpOnly cookies
- **Implementation**: `HttpOnly=true` on all session cookies

### Session Fixation
- **Mitigation**: New session on login
- **Implementation**: New session ID generated on OIDC callback

### Token Theft
- **Mitigation**: Token rotation, short TTLs
- **Implementation**: Refresh token rotation, 15-minute access tokens

### Authorization Code Interception
- **Mitigation**: PKCE
- **Implementation**: Code verifier/challenge exchange

### Replay Attacks
- **Mitigation**: One-time use tokens, expiration
- **Implementation**: Login attempts deleted after use, token expiration

## Monitoring and Alerts

### Key Metrics to Monitor

1. **Failed Login Attempts**:
   ```sql
   SELECT COUNT(*) FROM audit_log 
   WHERE event_type = 'login_failed' 
   AND created_at > NOW() - INTERVAL '1 hour';
   ```

2. **Token Refresh Rate**:
   ```sql
   SELECT COUNT(*) FROM audit_log 
   WHERE event_type = 'token_refresh' 
   AND created_at > NOW() - INTERVAL '1 hour';
   ```

3. **Session Revocations**:
   ```sql
   SELECT COUNT(*) FROM sessions 
   WHERE revoked_at IS NOT NULL 
   AND revoked_at > NOW() - INTERVAL '1 hour';
   ```

### Alert Conditions

- Multiple failed login attempts from same IP
- Unusual token refresh patterns
- Session revocation spikes
- Expired login attempts not cleaned up

## Migration Guide

### Running Migrations

```bash
# Run OIDC hardening migration
psql -d neurondesk -f NeuronDesktop/api/migrations/009_oidc_hardening.sql
```

### Verifying Migration

```sql
-- Check login_attempts table has redirect_uri
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'login_attempts' AND column_name = 'redirect_uri';

-- Check cleanup function exists
SELECT routine_name 
FROM information_schema.routines 
WHERE routine_name = 'cleanup_expired_login_attempts';
```

## Troubleshooting

### Issue: Login attempts not persisting

**Solution**: Verify database connection and table exists:
```sql
SELECT * FROM login_attempts LIMIT 1;
```

### Issue: Cookies not being set

**Solution**: Check cookie domain and secure settings:
```go
// Verify cookie settings in session manager
cookieSecure := os.Getenv("SESSION_COOKIE_SECURE") == "true"
```

### Issue: Token refresh failing

**Solution**: Check refresh token expiration:
```sql
SELECT * FROM refresh_tokens 
WHERE token_hash = '...' 
AND expires_at > NOW();
```

## References

- [OWASP Session Management](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- [OAuth 2.0 Security Best Practices](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-security-topics)
- [PKCE RFC 7636](https://tools.ietf.org/html/rfc7636)







