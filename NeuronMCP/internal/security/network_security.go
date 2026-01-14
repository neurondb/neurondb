/*-------------------------------------------------------------------------
 *
 * network_security.go
 *    Network security features (TLS, IP whitelisting, etc.)
 *
 * Implements network security features as specified in Phase 2.1.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/security/network_security.go
 *
 *-------------------------------------------------------------------------
 */

package security

import (
	"context"
	"fmt"
	"net"
	"strings"
)

/* IPFilter manages IP whitelisting and blacklisting */
type IPFilter struct {
	whitelist []*net.IPNet
	blacklist []*net.IPNet
}

/* NewIPFilter creates a new IP filter */
func NewIPFilter() *IPFilter {
	return &IPFilter{
		whitelist: []*net.IPNet{},
		blacklist: []*net.IPNet{},
	}
}

/* AddWhitelist adds an IP or CIDR to whitelist */
func (f *IPFilter) AddWhitelist(ipOrCIDR string) error {
	_, ipNet, err := net.ParseCIDR(ipOrCIDR)
	if err != nil {
		/* Try as single IP */
		ip := net.ParseIP(ipOrCIDR)
		if ip == nil {
			return fmt.Errorf("invalid IP or CIDR: %s", ipOrCIDR)
		}
		/* Convert to /32 or /128 */
		if ip.To4() != nil {
			_, ipNet, err = net.ParseCIDR(ipOrCIDR + "/32")
		} else {
			_, ipNet, err = net.ParseCIDR(ipOrCIDR + "/128")
		}
		if err != nil {
			return err
		}
	}
	f.whitelist = append(f.whitelist, ipNet)
	return nil
}

/* AddBlacklist adds an IP or CIDR to blacklist */
func (f *IPFilter) AddBlacklist(ipOrCIDR string) error {
	_, ipNet, err := net.ParseCIDR(ipOrCIDR)
	if err != nil {
		/* Try as single IP */
		ip := net.ParseIP(ipOrCIDR)
		if ip == nil {
			return fmt.Errorf("invalid IP or CIDR: %s", ipOrCIDR)
		}
		/* Convert to /32 or /128 */
		if ip.To4() != nil {
			_, ipNet, err = net.ParseCIDR(ipOrCIDR + "/32")
		} else {
			_, ipNet, err = net.ParseCIDR(ipOrCIDR + "/128")
		}
		if err != nil {
			return err
		}
	}
	f.blacklist = append(f.blacklist, ipNet)
	return nil
}

/* IsAllowed checks if an IP is allowed */
func (f *IPFilter) IsAllowed(ipStr string) bool {
	ip := net.ParseIP(ipStr)
	if ip == nil {
		return false
	}

	/* Check blacklist first */
	for _, ipNet := range f.blacklist {
		if ipNet.Contains(ip) {
			return false
		}
	}

	/* If whitelist is empty, allow all (except blacklisted) */
	if len(f.whitelist) == 0 {
		return true
	}

	/* Check whitelist */
	for _, ipNet := range f.whitelist {
		if ipNet.Contains(ip) {
			return true
		}
	}

	return false
}

/* GetClientIP extracts client IP from context */
func GetClientIP(ctx context.Context) string {
	/* Try to get from context */
	if ip, ok := ctx.Value("client_ip").(string); ok {
		return ip
	}
	return ""
}

/* TLSConfig represents TLS configuration */
type TLSConfig struct {
	MinVersion     string /* "1.2", "1.3" */
	MaxVersion     string
	CertificatePin []string /* Certificate fingerprints */
	RequireClientCert bool
}

/* ValidateTLSVersion validates TLS version */
func ValidateTLSVersion(version string) bool {
	validVersions := []string{"1.2", "1.3"}
	for _, v := range validVersions {
		if version == v {
			return true
		}
	}
	return false
}

/* CertificatePinning verifies certificate pinning */
type CertificatePinner struct {
	pinnedCerts map[string]bool /* fingerprint -> true */
}

/* NewCertificatePinner creates a new certificate pinner */
func NewCertificatePinner() *CertificatePinner {
	return &CertificatePinner{
		pinnedCerts: make(map[string]bool),
	}
}

/* AddPinnedCert adds a pinned certificate */
func (p *CertificatePinner) AddPinnedCert(fingerprint string) {
	p.pinnedCerts[strings.ToLower(fingerprint)] = true
}

/* IsPinned checks if a certificate is pinned */
func (p *CertificatePinner) IsPinned(fingerprint string) bool {
	return p.pinnedCerts[strings.ToLower(fingerprint)]
}



