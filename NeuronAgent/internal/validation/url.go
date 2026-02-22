/*-------------------------------------------------------------------------
 *
 * url.go
 *    SSRF-safe URL validation for NeuronAgent
 *
 * Validates URL scheme and blocks private/internal IP ranges to prevent
 * SSRF and DNS rebinding attacks.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/validation/url.go
 *
 *-------------------------------------------------------------------------
 */

package validation

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
	"time"
)

/* Allowed URL schemes for outbound HTTP tool */
var allowedSchemes = map[string]bool{
	"http":  true,
	"https": true,
}

/* ValidateURLForSSRF parses the URL, allows only http/https, and ensures the host
 * does not resolve to a private or internal IP (prevents SSRF and DNS rebinding).
 * If resolver is nil, uses default resolver with a short timeout. */
func ValidateURLForSSRF(rawURL string, resolver *net.Resolver) error {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return fmt.Errorf("invalid URL: %w", err)
	}
	scheme := strings.ToLower(u.Scheme)
	if !allowedSchemes[scheme] {
		return fmt.Errorf("URL scheme not allowed: %q (only http and https are permitted)", u.Scheme)
	}
	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("URL has no host")
	}
	/* Reject explicit private/internal IPs in the URL */
	if ip := net.ParseIP(host); ip != nil {
		if isPrivateOrInternal(ip) {
			return fmt.Errorf("URL host is a private or internal IP address")
		}
		return nil
	}
	/* Resolve hostname and check all resolved IPs are public */
	if resolver == nil {
		resolver = net.DefaultResolver
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ips, err := resolver.LookupIP(ctx, "ip", host)
	if err != nil {
		return fmt.Errorf("failed to resolve host %q: %w", host, err)
	}
	for _, ip := range ips {
		/* Normalize to IPv4 for comparison */
		if ip4 := ip.To4(); ip4 != nil {
			ip = ip4
		}
		if isPrivateOrInternal(ip) {
			return fmt.Errorf("URL host %q resolves to private/internal IP %s (possible DNS rebinding)", host, ip)
		}
	}
	return nil
}

/* isPrivateOrInternal returns true for private, loopback, link-local, or reserved ranges */
func isPrivateOrInternal(ip net.IP) bool {
	if ip4 := ip.To4(); ip4 != nil {
		/* 127.0.0.0/8 loopback */
		if ip4[0] == 127 {
			return true
		}
		/* 10.0.0.0/8 */
		if ip4[0] == 10 {
			return true
		}
		/* 172.16.0.0/12 */
		if ip4[0] == 172 && ip4[1] >= 16 && ip4[1] <= 31 {
			return true
		}
		/* 192.168.0.0/16 */
		if ip4[0] == 192 && ip4[1] == 168 {
			return true
		}
		/* 169.254.0.0/16 link-local */
		if ip4[0] == 169 && ip4[1] == 254 {
			return true
		}
		/* 0.0.0.0/8 */
		if ip4[0] == 0 {
			return true
		}
		return false
	}
	/* IPv6: loopback, link-local, unique local, unspecified */
	if ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() ||
		ip.IsUnspecified() || isIPv6Private(ip) {
		return true
	}
	return false
}

func isIPv6Private(ip net.IP) bool {
	/* fc00::/7 unique local */
	if len(ip) >= 2 && ip[0] == 0xfc {
		return true
	}
	if len(ip) >= 2 && ip[0] == 0xfd {
		return true
	}
	return false
}
