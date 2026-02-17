/* Fuzz test for SQL query validation (go test -fuzz=FuzzValidateSQLQuery -fuzztime=30s) */

package validation

import (
	"testing"
)

func FuzzValidateSQLQuery(f *testing.F) {
	seeds := []string{
		"SELECT 1",
		"SELECT * FROM t",
		"DROP TABLE t",
		"INSERT INTO t VALUES (1)",
		"SELECT 1; DELETE FROM t",
		"",
		"  \n  SELECT a FROM b  ",
		"EXPLAIN SELECT 1",
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, query string) {
		_ = ValidateSQLQuery(query, AllowReadOnly)
		_ = ValidateSQLQuery(query, AllowSelectOnly)
	})
}
