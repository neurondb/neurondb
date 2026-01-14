/*-------------------------------------------------------------------------
 *
 * query.go
 *    SQL query builder for NeuronMCP
 *
 * Provides utilities for building SQL queries including SELECT statements
 * and vector search queries.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/database/query.go
 *
 *-------------------------------------------------------------------------
 */

package database

import (
	"fmt"
	"strings"
)

/* QueryBuilder provides utilities for building SQL queries */
type QueryBuilder struct{}

/* Select builds a SELECT query */
func (qb *QueryBuilder) Select(table string, columns []string, where map[string]interface{}, orderBy *OrderBy, limit, offset *int) (string, []interface{}) {
	if len(columns) == 0 {
		columns = []string{"*"}
	}

	var params []interface{}
	paramIndex := 1

	selectClause := strings.Join(columns, ", ")
	fromClause := EscapeIdentifier(table)

	var whereClause string
	if len(where) > 0 {
		var conditions []string
		for key, value := range where {
			escapedKey := EscapeIdentifier(key)
			conditions = append(conditions, fmt.Sprintf("%s = $%d", escapedKey, paramIndex))
			params = append(params, value)
			paramIndex++
		}
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	var orderByClause string
	if orderBy != nil {
		orderByClause = fmt.Sprintf("ORDER BY %s %s", EscapeIdentifier(orderBy.Column), orderBy.Direction)
	}

	var limitClause string
	if limit != nil {
		limitClause = fmt.Sprintf("LIMIT $%d", paramIndex)
		params = append(params, *limit)
		paramIndex++
	}

	var offsetClause string
	if offset != nil {
		offsetClause = fmt.Sprintf("OFFSET $%d", paramIndex)
		params = append(params, *offset)
	}

	parts := []string{
		"SELECT " + selectClause,
		"FROM " + fromClause,
		whereClause,
		orderByClause,
		limitClause,
		offsetClause,
	}

	var nonEmptyParts []string
	for _, part := range parts {
		if part != "" {
			nonEmptyParts = append(nonEmptyParts, part)
		}
	}

	query := strings.Join(nonEmptyParts, " ")
	return query, params
}

/* OrderBy represents an ORDER BY clause */
type OrderBy struct {
	Column    string
	Direction string
}

/* VectorSearch builds a vector search query */
func (qb *QueryBuilder) VectorSearch(table, vectorColumn string, queryVector []float32, distanceMetric string, limit int, additionalColumns []string, minkowskiP *float64) (string, []interface{}) {
	if len(queryVector) == 0 {
		return "", nil
	}

	var params []interface{}
	paramIndex := 1

	vectorStr := formatVector(queryVector)
	params = append(params, vectorStr)
	vectorParamIndex := paramIndex
	paramIndex++

	tableAlias := "t"
	params = append(params, limit)
	limitParamIndex := paramIndex

	/* Use subquery to cast vector to text, then calculate distance in outer query using cast back to vector */
	/* This ensures vector is scannable while allowing distance calculation */
	subquerySelect := []string{}
	if len(additionalColumns) > 0 {
		for _, col := range additionalColumns {
			subquerySelect = append(subquerySelect, EscapeIdentifier(col))
		}
		subquerySelect = append(subquerySelect, fmt.Sprintf("%s::text AS %s", EscapeIdentifier(vectorColumn), EscapeIdentifier(vectorColumn)))
	} else {
		/* For *, we need to explicitly cast the vector column to text */
		/* PostgreSQL doesn't support * EXCEPT column, so we'll handle it differently */
		/* Select all columns with vector cast to text */
		subquerySelect = append(subquerySelect, "*")
		/* Note: This creates a duplicate, but we'll handle it by selecting explicitly in outer query */
	}
	
	subquery := fmt.Sprintf("(SELECT %s FROM %s) AS %s",
		strings.Join(subquerySelect, ", "),
		EscapeIdentifier(table),
		tableAlias)

	/* Build final SELECT - calculate distance using vector cast, select all columns */
	/* Distance calculation: cast the text vector back to vector type for distance calc */
	distanceExprWithCast := ""
	switch distanceMetric {
	case "cosine":
		distanceExprWithCast = fmt.Sprintf("(%s.%s::vector <=> $%d::vector) AS distance", tableAlias, EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "inner_product":
		distanceExprWithCast = fmt.Sprintf("(%s.%s::vector <#> $%d::vector) AS distance", tableAlias, EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "l1":
		distanceExprWithCast = fmt.Sprintf("vector_l1_distance(%s.%s::vector, $%d::vector) AS distance", tableAlias, EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "hamming":
		distanceExprWithCast = fmt.Sprintf("vector_hamming_distance(%s.%s::vector, $%d::vector) AS distance", tableAlias, EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "chebyshev":
		distanceExprWithCast = fmt.Sprintf("vector_chebyshev_distance(%s.%s::vector, $%d::vector) AS distance", tableAlias, EscapeIdentifier(vectorColumn), vectorParamIndex)
	case "minkowski":
		p := 2.0
		if minkowskiP != nil {
			p = *minkowskiP
		}
		params = append(params, p)
		pParamIndex := paramIndex
		paramIndex++
		distanceExprWithCast = fmt.Sprintf("vector_minkowski_distance(%s.%s::vector, $%d::vector, $%d::double precision) AS distance", tableAlias, EscapeIdentifier(vectorColumn), vectorParamIndex, pParamIndex)
	default:
		distanceExprWithCast = fmt.Sprintf("(%s.%s::vector <-> $%d::vector) AS distance", tableAlias, EscapeIdentifier(vectorColumn), vectorParamIndex)
	}

	selectColumns := []string{}
	if len(additionalColumns) > 0 {
		for _, col := range additionalColumns {
			selectColumns = append(selectColumns, fmt.Sprintf("%s.%s", tableAlias, EscapeIdentifier(col)))
		}
		selectColumns = append(selectColumns, fmt.Sprintf("%s.%s", tableAlias, EscapeIdentifier(vectorColumn)))
	} else {
		/* Select all columns from subquery (vector is already text) */
		selectColumns = append(selectColumns, fmt.Sprintf("%s.*", tableAlias))
	}
	selectColumns = append(selectColumns, distanceExprWithCast)

	selectClause := strings.Join(selectColumns, ", ")
	query := fmt.Sprintf(
		"SELECT %s FROM %s ORDER BY distance ASC LIMIT $%d",
		selectClause,
		subquery,
		limitParamIndex,
	)

	return query, params
}

/* formatVector formats a float32 slice as a PostgreSQL vector string */
func formatVector(vec []float32) string {
	var parts []string
	for _, v := range vec {
		parts = append(parts, fmt.Sprintf("%g", v))
	}
	return "[" + strings.Join(parts, ",") + "]"
}

