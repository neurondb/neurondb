/*-------------------------------------------------------------------------
 *
 * s3.go
 *    S3-compatible object storage connector implementation
 *
 * Provides S3-compatible storage integration (AWS S3, MinIO, etc.).
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/connectors/s3.go
 *
 *-------------------------------------------------------------------------
 */

package connectors

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

/* S3Connector implements ReadWriteConnector for S3 */
type S3Connector struct {
	endpoint  string
	region    string
	bucket    string
	accessKey string
	secretKey string
}

/* NewS3Connector creates a new S3 connector */
func NewS3Connector(config Config) (*S3Connector, error) {
	if config.Metadata == nil {
		return nil, fmt.Errorf("S3 config requires bucket in metadata")
	}

	bucket, ok := config.Metadata["bucket"].(string)
	if !ok || bucket == "" {
		return nil, fmt.Errorf("S3 bucket is required")
	}

	region := "us-east-1"
	if r, ok := config.Metadata["region"].(string); ok {
		region = r
	}

	accessKey := config.Token
	if ak, ok := config.Metadata["access_key"].(string); ok {
		accessKey = ak
	}

	secretKey := ""
	if sk, ok := config.Metadata["secret_key"].(string); ok {
		secretKey = sk
	}

	return &S3Connector{
		endpoint:  config.Endpoint,
		region:    region,
		bucket:    bucket,
		accessKey: accessKey,
		secretKey: secretKey,
	}, nil
}

/* Type returns the connector type */
func (s *S3Connector) Type() string {
	return "s3"
}

/* Connect establishes connection */
func (s *S3Connector) Connect(ctx context.Context) error {
	/* Test connection by listing bucket (with limit 1) */
	_, err := s.List(ctx, "")
	if err != nil {
		return fmt.Errorf("S3 connection test failed: %w", err)
	}
	return nil
}

/* Close closes the connection */
func (s *S3Connector) Close() error {
	return nil
}

/* Health checks connection health */
func (s *S3Connector) Health(ctx context.Context) error {
	return s.Connect(ctx)
}

/* Read reads an object from S3 */
func (s *S3Connector) Read(ctx context.Context, path string) (io.Reader, error) {
	/* Build S3 URL */
	url := s.buildURL(path)
	
	/* Create request */
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create S3 request: %w", err)
	}

	/* Sign request */
	s.signRequest(req, "GET", path, "", nil)

	/* Execute request */
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("S3 read request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("S3 read failed: status_code=%d", resp.StatusCode)
	}

	return resp.Body, nil
}

/* Write writes an object to S3 */
func (s *S3Connector) Write(ctx context.Context, path string, data io.Reader) error {
	/* Read all data */
	bodyBytes, err := io.ReadAll(data)
	if err != nil {
		return fmt.Errorf("failed to read data: %w", err)
	}

	/* Build S3 URL */
	url := s.buildURL(path)

	/* Create request */
	req, err := http.NewRequestWithContext(ctx, "PUT", url, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("failed to create S3 request: %w", err)
	}

	/* Set content type */
	req.Header.Set("Content-Type", "application/octet-stream")

	/* Sign request */
	s.signRequest(req, "PUT", path, "", bodyBytes)

	/* Execute request */
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("S3 write request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != 200 {
		return fmt.Errorf("S3 write failed: status_code=%d", resp.StatusCode)
	}

	return nil
}

/* List lists objects in S3 bucket */
func (s *S3Connector) List(ctx context.Context, path string) ([]string, error) {
	/* Build S3 URL with list query */
	url := s.buildURL("")
	if path != "" {
		url += "?prefix=" + path
	}

	/* Create request */
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create S3 request: %w", err)
	}

	/* Sign request */
	s.signRequest(req, "GET", "", "", nil)

	/* Execute request */
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("S3 list request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("S3 list failed: status_code=%d", resp.StatusCode)
	}

	/* Parse XML response (simplified - in production use XML parser) */
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read S3 response: %w", err)
	}

	/* Simple XML parsing for object keys */
	var keys []string
	bodyStr := string(body)
	startTag := "<Key>"
	endTag := "</Key>"
	startIdx := 0
	for {
		idx := strings.Index(bodyStr[startIdx:], startTag)
		if idx == -1 {
			break
		}
		keyStart := startIdx + idx + len(startTag)
		endIdx := strings.Index(bodyStr[keyStart:], endTag)
		if endIdx == -1 {
			break
		}
		key := bodyStr[keyStart : keyStart+endIdx]
		keys = append(keys, key)
		startIdx = keyStart + endIdx + len(endTag)
	}

	return keys, nil
}

/* buildURL builds S3 URL for an object */
func (s *S3Connector) buildURL(path string) string {
	if s.endpoint != "" {
		/* Custom endpoint (MinIO, etc.) */
		if strings.HasSuffix(s.endpoint, "/") {
			return s.endpoint + s.bucket + "/" + path
		}
		return s.endpoint + "/" + s.bucket + "/" + path
	}
	/* AWS S3 */
	if s.region == "" {
		s.region = "us-east-1"
	}
	return fmt.Sprintf("https://%s.s3.%s.amazonaws.com/%s", s.bucket, s.region, path)
}

/* signRequest signs an S3 request using AWS Signature Version 4 */
func (s *S3Connector) signRequest(req *http.Request, method, path, contentType string, body []byte) {
	if s.accessKey == "" || s.secretKey == "" {
		/* No credentials, skip signing (for public buckets) */
		return
	}

	/* Simplified signing - in production use proper AWS SigV4 */
	/* For now, use basic auth or skip if credentials not available */
	timestamp := time.Now().UTC().Format("20060102T150405Z")
	dateStamp := timestamp[:8]

	/* Create string to sign */
	stringToSign := method + "\n\n" + contentType + "\n" + timestamp + "\n/" + s.bucket + "/" + path

	/* Create signing key */
	kDate := s.hmacSHA256([]byte("AWS4"+s.secretKey), dateStamp)
	kRegion := s.hmacSHA256(kDate, s.region)
	kService := s.hmacSHA256(kRegion, "s3")
	kSigning := s.hmacSHA256(kService, "aws4_request")

	/* Sign */
	signature := hex.EncodeToString(s.hmacSHA256(kSigning, stringToSign))

	/* Set authorization header */
	authHeader := fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s/%s/s3/aws4_request, SignedHeaders=host;x-amz-date, Signature=%s",
		s.accessKey, dateStamp, s.region, signature)
	req.Header.Set("Authorization", authHeader)
	req.Header.Set("X-Amz-Date", timestamp)
}

/* hmacSHA256 computes HMAC-SHA256 */
func (s *S3Connector) hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}
