package middleware

import (
	"html"
	"net/http"
	"strings"
)

/* XSSSanitizeMiddleware sanitizes user input to prevent XSS attacks */
func XSSSanitizeMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			/* Sanitize query parameters */
			query := r.URL.Query()
			for key, values := range query {
				for i, value := range values {
					query[key][i] = sanitizeString(value)
				}
			}
			r.URL.RawQuery = query.Encode()

			/* Sanitize form values */
			if err := r.ParseForm(); err == nil {
				for key, values := range r.PostForm {
					for i, value := range values {
						r.PostForm[key][i] = sanitizeString(value)
					}
				}
			}

			next.ServeHTTP(w, r)
		})
	}
}

/* sanitizeString removes potentially dangerous HTML/script content */
func sanitizeString(s string) string {
	/* HTML escape */
	s = html.EscapeString(s)

	/* Remove script tags and event handlers */
	dangerousPatterns := []string{
		"<script",
		"</script>",
		"javascript:",
		"onerror=",
		"onclick=",
		"onload=",
		"onmouseover=",
	}

	for _, pattern := range dangerousPatterns {
		s = strings.ReplaceAll(strings.ToLower(s), strings.ToLower(pattern), "")
	}

	return s
}

/* SanitizeString is a public function for sanitizing strings */
func SanitizeString(s string) string {
	return sanitizeString(s)
}








