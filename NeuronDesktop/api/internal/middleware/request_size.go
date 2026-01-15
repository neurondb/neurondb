package middleware

import (
	"io"
	"net/http"
)

/* RequestSizeMiddleware limits the size of request bodies */
func RequestSizeMiddleware(maxSize int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			/* Limit request body size */
			r.Body = http.MaxBytesReader(w, r.Body, maxSize)

			next.ServeHTTP(w, r)
		})
	}
}

/* ReadBodyWithLimit reads a request body with a size limit */
func ReadBodyWithLimit(r *http.Request, maxSize int64) ([]byte, error) {
	limitedReader := io.LimitReader(r.Body, maxSize)
	return io.ReadAll(limitedReader)
}








