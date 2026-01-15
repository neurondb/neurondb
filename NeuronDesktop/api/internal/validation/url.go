package validation

import (
	"fmt"
	"net/url"
	"strings"
)

/* ValidateURL validates a URL string */
func ValidateURL(urlStr string) bool {
	if urlStr == "" {
		return false
	}

	urlStr = strings.TrimSpace(urlStr)

	parsed, err := url.Parse(urlStr)
	if err != nil {
		return false
	}

	/* Must have a scheme */
	if parsed.Scheme == "" {
		return false
	}

	/* Must be http or https */
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return false
	}

	/* Must have a host */
	if parsed.Host == "" {
		return false
	}

	return true
}

/* ValidateURLRequired validates a URL and ensures it's not empty */
func ValidateURLRequired(urlStr, fieldName string) error {
	if urlStr == "" {
		return fmt.Errorf("%s is required and cannot be empty", fieldName)
	}

	if !ValidateURL(urlStr) {
		return fmt.Errorf("%s is not a valid URL", fieldName)
	}

	return nil
}








