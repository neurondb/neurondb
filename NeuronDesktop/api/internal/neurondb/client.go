package neurondb

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
)

/* Client provides access to NeuronDB via Postgres */
type Client struct {
	db *sql.DB
}

/* NewClient creates a new NeuronDB client */
func NewClient(dsn string) (*Client, error) {
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &Client{db: db}, nil
}

/* Close closes the database connection */
func (c *Client) Close() error {
	return c.db.Close()
}

/* CollectionInfo represents a collection (table) in NeuronDB */
type CollectionInfo struct {
	Name      string            `json:"name"`
	Schema    string            `json:"schema"`
	VectorCol string            `json:"vector_col,omitempty"`
	Indexes   []IndexInfo       `json:"indexes,omitempty"`
	RowCount  int64             `json:"row_count,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

/* IndexInfo represents an index on a collection */
type IndexInfo struct {
	Name       string `json:"name"`
	Type       string `json:"type"`
	Definition string `json:"definition"`
	Size       string `json:"size,omitempty"`
}

/* ListCollections lists all tables that might contain vectors */
func (c *Client) ListCollections(ctx context.Context) ([]CollectionInfo, error) {
	query := `
		SELECT 
			schemaname,
			tablename,
			(SELECT COUNT(*) FROM information_schema.columns 
			 WHERE table_schema = t.schemaname 
			 AND table_name = t.tablename 
			 AND udt_name = 'vector') as has_vector
		FROM pg_tables t
		WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
		AND (SELECT COUNT(*) FROM information_schema.columns 
		     WHERE table_schema = t.schemaname 
		     AND table_name = t.tablename 
		     AND udt_name = 'vector') > 0
		ORDER BY schemaname, tablename
	`

	rows, err := c.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query collections: %w", err)
	}
	defer rows.Close()

	var collections []CollectionInfo
	for rows.Next() {
		var schema, table string
		var hasVector bool
		if err := rows.Scan(&schema, &table, &hasVector); err != nil {
			continue
		}

		if !hasVector {
			continue
		}

		vectorCol, err := c.getVectorColumn(ctx, schema, table)
		if err != nil {
			continue
		}

		indexes, _ := c.getIndexes(ctx, schema, table)

		var rowCount int64
		if err := validateIdentifier(schema); err != nil {
			continue
		}
		if err := validateIdentifier(table); err != nil {
			continue
		}
		countQuery := fmt.Sprintf(`SELECT COUNT(*) FROM %s.%s`,
			quoteIdentifier(schema), quoteIdentifier(table))
		c.db.QueryRowContext(ctx, countQuery).Scan(&rowCount)

		collections = append(collections, CollectionInfo{
			Name:      table,
			Schema:    schema,
			VectorCol: vectorCol,
			Indexes:   indexes,
			RowCount:  rowCount,
		})
	}

	return collections, nil
}

/* getVectorColumn finds the vector column in a table */
func (c *Client) getVectorColumn(ctx context.Context, schema, table string) (string, error) {
	query := `
		SELECT column_name
		FROM information_schema.columns
		WHERE table_schema = $1
		AND table_name = $2
		AND udt_name = 'vector'
		LIMIT 1
	`

	var colName string
	err := c.db.QueryRowContext(ctx, query, schema, table).Scan(&colName)
	return colName, err
}

/* getIndexes gets indexes for a table */
func (c *Client) getIndexes(ctx context.Context, schema, table string) ([]IndexInfo, error) {
	query := `
		SELECT 
			i.indexname,
			am.amname as index_type,
			pg_get_indexdef(i.indexrelid) as definition,
			pg_size_pretty(pg_relation_size(i.indexrelid)) as size
		FROM pg_indexes i
		JOIN pg_class c ON c.relname = i.indexname
		JOIN pg_am am ON am.oid = c.relam
		WHERE i.schemaname = $1
		AND i.tablename = $2
		ORDER BY i.indexname
	`

	rows, err := c.db.QueryContext(ctx, query, schema, table)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var indexes []IndexInfo
	for rows.Next() {
		var idx IndexInfo
		if err := rows.Scan(&idx.Name, &idx.Type, &idx.Definition, &idx.Size); err != nil {
			continue
		}
		indexes = append(indexes, idx)
	}

	return indexes, nil
}

/* SearchRequest represents a vector search request */
type SearchRequest struct {
	Collection   string                 `json:"collection"`
	Schema       string                 `json:"schema,omitempty"`
	QueryVector  []float32              `json:"query_vector"`
	QueryText    string                 `json:"query_text,omitempty"` // For hybrid search
	Limit        int                    `json:"limit,omitempty"`
	Filter       map[string]interface{} `json:"filter,omitempty"`
	DistanceType string                 `json:"distance_type,omitempty"` // l2, cosine, inner_product, l1, hamming, chebyshev, minkowski
	MinkowskiP   *float64               `json:"minkowski_p,omitempty"`   // p parameter for Minkowski distance
}

/* SearchResult represents a single search result */
type SearchResult struct {
	ID       interface{}            `json:"id"`
	Score    float64                `json:"score"`
	Distance float64                `json:"distance"`
	Data     map[string]interface{} `json:"data"`
}

/* Search performs a vector search */
func (c *Client) Search(ctx context.Context, req SearchRequest) ([]SearchResult, error) {
	if req.Limit <= 0 {
		req.Limit = 10
	}
	if req.Schema == "" {
		req.Schema = "public"
	}
	if req.DistanceType == "" {
		req.DistanceType = "cosine"
	}

	/* Validate identifiers (prevent SQL injection) */
	if err := validateIdentifier(req.Schema); err != nil {
		return nil, fmt.Errorf("invalid schema name: %w", err)
	}
	if err := validateIdentifier(req.Collection); err != nil {
		return nil, fmt.Errorf("invalid collection name: %w", err)
	}

	vectorCol, err := c.getVectorColumn(ctx, req.Schema, req.Collection)
	if err != nil {
		return nil, fmt.Errorf("failed to find vector column: %w", err)
	}

	/* Validate vector column name */
	if err := validateIdentifier(vectorCol); err != nil {
		return nil, fmt.Errorf("invalid vector column name: %w", err)
	}

	/* Convert query vector to SQL array format (safe - numeric array) */
	vectorStr := formatVector(req.QueryVector)

	/* Build distance expression (safe - hardcoded values) */
	var distanceExpr string
	var orderByExpr string
	var queryParams []interface{}
	queryParams = append(queryParams, vectorStr)
	paramIndex := 2 // $1 is vector, $2+ for other params

	switch req.DistanceType {
	case "l2":
		distanceExpr = fmt.Sprintf("(%s <-> $1::vector) as distance", quoteIdentifier(vectorCol))
		orderByExpr = fmt.Sprintf("%s <-> $1::vector", quoteIdentifier(vectorCol))
	case "cosine":
		distanceExpr = fmt.Sprintf("(%s <=> $1::vector) as distance", quoteIdentifier(vectorCol))
		orderByExpr = fmt.Sprintf("%s <=> $1::vector", quoteIdentifier(vectorCol))
	case "inner_product":
		distanceExpr = fmt.Sprintf("(%s <#> $1::vector) as distance", quoteIdentifier(vectorCol))
		orderByExpr = fmt.Sprintf("%s <#> $1::vector", quoteIdentifier(vectorCol))
	case "l1":
		distanceExpr = fmt.Sprintf("vector_l1_distance(%s, $1::vector) as distance", quoteIdentifier(vectorCol))
		orderByExpr = fmt.Sprintf("vector_l1_distance(%s, $1::vector)", quoteIdentifier(vectorCol))
	case "hamming":
		distanceExpr = fmt.Sprintf("vector_hamming_distance(%s, $1::vector) as distance", quoteIdentifier(vectorCol))
		orderByExpr = fmt.Sprintf("vector_hamming_distance(%s, $1::vector)", quoteIdentifier(vectorCol))
	case "chebyshev":
		distanceExpr = fmt.Sprintf("vector_chebyshev_distance(%s, $1::vector) as distance", quoteIdentifier(vectorCol))
		orderByExpr = fmt.Sprintf("vector_chebyshev_distance(%s, $1::vector)", quoteIdentifier(vectorCol))
	case "minkowski":
		p := 2.0
		if req.MinkowskiP != nil {
			p = *req.MinkowskiP
		}
		queryParams = append(queryParams, p)
		distanceExpr = fmt.Sprintf("vector_minkowski_distance(%s, $1::vector, $%d::double precision) as distance", quoteIdentifier(vectorCol), paramIndex)
		orderByExpr = fmt.Sprintf("vector_minkowski_distance(%s, $1::vector, $%d::double precision)", quoteIdentifier(vectorCol), paramIndex)
		paramIndex++
	default:
		/* Default to cosine */
		distanceExpr = fmt.Sprintf("(%s <=> $1::vector) as distance", quoteIdentifier(vectorCol))
		orderByExpr = fmt.Sprintf("%s <=> $1::vector", quoteIdentifier(vectorCol))
	}

	queryParams = append(queryParams, req.Limit)
	limitParamIndex := paramIndex

	/* Build query with validated identifiers */
	/* Note: vectorStr is generated from numeric array, so safe to interpolate */
	/* Identifiers are validated above, so safe to use in query */
	tableName := fmt.Sprintf("%s.%s", quoteIdentifier(req.Schema), quoteIdentifier(req.Collection))
	query := fmt.Sprintf(`
		SELECT *, 
			%s
		FROM %s
		ORDER BY %s
		LIMIT $%d
	`, distanceExpr, tableName, orderByExpr, limitParamIndex)

	/* Execute query (all parameters are safe) */
	rows, err := c.db.QueryContext(ctx, query, queryParams...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search: %w", err)
	}
	defer rows.Close()

	/* Get column names */
	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	var results []SearchResult
	for rows.Next() {
		/* Create slice to hold values */
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			continue
		}

		/* Build result map */
		result := SearchResult{
			Data: make(map[string]interface{}),
		}

		for i, col := range columns {
			val := values[i]

			if col == "distance" {
				if d, ok := val.(float64); ok {
					result.Distance = d
					/* Convert distance to score (1 / (1 + distance) for cosine) */
					if req.DistanceType == "cosine" {
						result.Score = 1.0 - d
					} else {
						result.Score = 1.0 / (1.0 + d)
					}
				}
			} else if col == "id" || strings.HasSuffix(col, "_id") {
				result.ID = val
			} else {
				/* Convert to JSON-serializable type */
				if bytes, ok := val.([]byte); ok {
					var jsonVal interface{}
					if err := json.Unmarshal(bytes, &jsonVal); err == nil {
						result.Data[col] = jsonVal
					} else {
						result.Data[col] = string(bytes)
					}
				} else {
					result.Data[col] = val
				}
			}
		}

		results = append(results, result)
	}

	return results, nil
}

/* ExecuteSQL executes a SQL query with guardrails */
func (c *Client) ExecuteSQL(ctx context.Context, query string) (interface{}, error) {
	queryUpper := strings.ToUpper(strings.TrimSpace(query))
	dangerous := []string{"DROP", "TRUNCATE", "DELETE", "UPDATE", "ALTER", "CREATE", "GRANT", "REVOKE"}

	for _, keyword := range dangerous {
		if strings.Contains(queryUpper, keyword) {
			return nil, fmt.Errorf("dangerous SQL operation not allowed: %s", keyword)
		}
	}

	/* Only allow SELECT statements */
	if !strings.HasPrefix(queryUpper, "SELECT") {
		return nil, fmt.Errorf("only SELECT queries are allowed")
	}

	rows, err := c.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("SQL execution failed: %w", err)
	}
	defer rows.Close()

	columns, err := rows.Columns()
	if err != nil {
		return nil, fmt.Errorf("failed to get columns: %w", err)
	}

	var results []map[string]interface{}
	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			continue
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			val := values[i]
			if bytes, ok := val.([]byte); ok {
				var jsonVal interface{}
				if err := json.Unmarshal(bytes, &jsonVal); err == nil {
					row[col] = jsonVal
				} else {
					row[col] = string(bytes)
				}
			} else {
				row[col] = val
			}
		}
		results = append(results, row)
	}

	return results, nil
}

/* ExecuteSQLFull executes any SQL query (full database access) */
/* This allows CREATE, INSERT, UPDATE, DELETE, DROP, etc. */
/* Use with extreme caution - no safety checks are performed */
func (c *Client) ExecuteSQLFull(ctx context.Context, query string) (interface{}, error) {
	queryUpper := strings.ToUpper(strings.TrimSpace(query))

	/* Check if it's a SELECT query (return results) */
	if strings.HasPrefix(queryUpper, "SELECT") {
		rows, err := c.db.QueryContext(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("SQL execution failed: %w", err)
		}
		defer rows.Close()

		columns, err := rows.Columns()
		if err != nil {
			return nil, fmt.Errorf("failed to get columns: %w", err)
		}

		/* Initialize as empty array instead of nil to ensure JSON serialization as [] */
		results := make([]map[string]interface{}, 0)
		for rows.Next() {
			values := make([]interface{}, len(columns))
			valuePtrs := make([]interface{}, len(columns))
			for i := range values {
				valuePtrs[i] = &values[i]
			}

			if err := rows.Scan(valuePtrs...); err != nil {
				continue
			}

			row := make(map[string]interface{})
			for i, col := range columns {
				val := values[i]
				if bytes, ok := val.([]byte); ok {
					var jsonVal interface{}
					if err := json.Unmarshal(bytes, &jsonVal); err == nil {
						row[col] = jsonVal
					} else {
						row[col] = string(bytes)
					}
				} else {
					row[col] = val
				}
			}
			results = append(results, row)
		}

		return results, nil
	}

	/* For non-SELECT queries (INSERT, UPDATE, DELETE, CREATE, etc.) */
	/* Execute the query and return affected rows count */
	result, err := c.db.ExecContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("SQL execution failed: %w", err)
	}

	rowsAffected, _ := result.RowsAffected()

	return map[string]interface{}{
		"rows_affected": rowsAffected,
		"message":       "Query executed successfully",
	}, nil
}

/* Helper functions */

/* validateIdentifier validates that an identifier is safe to use in SQL */
/* Only allows alphanumeric, underscore, and must start with letter/underscore */
var identifierRegex = regexp.MustCompile(`^[a-zA-Z_][a-zA-Z0-9_]*$`)

func validateIdentifier(name string) error {
	if name == "" {
		return fmt.Errorf("identifier cannot be empty")
	}
	if len(name) > 63 {
		return fmt.Errorf("identifier too long (max 63 chars)")
	}
	if !identifierRegex.MatchString(name) {
		return fmt.Errorf("identifier contains invalid characters: %s", name)
	}
	return nil
}

func quoteIdentifier(name string) string {
	return `"` + strings.ReplaceAll(name, `"`, `""`) + `"`
}

func formatVector(vec []float32) string {
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = fmt.Sprintf("%.6f", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}
