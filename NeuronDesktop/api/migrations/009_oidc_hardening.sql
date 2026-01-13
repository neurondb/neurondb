-- ============================================================================
-- OIDC Hardening Migration
-- ============================================================================
-- This migration adds redirect_uri to login_attempts table and ensures
-- proper cleanup of expired login attempts.
-- ============================================================================

-- Add redirect_uri column to login_attempts if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'login_attempts' AND column_name = 'redirect_uri'
    ) THEN
        ALTER TABLE login_attempts ADD COLUMN redirect_uri TEXT;
    END IF;
END $$;

-- Create index for cleanup queries
CREATE INDEX IF NOT EXISTS idx_login_attempts_expires_at_cleanup 
ON login_attempts(expires_at) WHERE expires_at < NOW();

-- Function to clean up expired login attempts (run periodically)
CREATE OR REPLACE FUNCTION cleanup_expired_login_attempts()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM login_attempts
    WHERE expires_at < NOW() - INTERVAL '1 hour';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Add comment
COMMENT ON FUNCTION cleanup_expired_login_attempts() IS 
'Cleans up expired login attempts older than 1 hour. Should be run periodically via cron or scheduled job.';







