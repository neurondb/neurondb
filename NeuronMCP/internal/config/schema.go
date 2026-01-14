/*-------------------------------------------------------------------------
 *
 * schema.go
 *    Database operations
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/config/schema.go
 *
 *-------------------------------------------------------------------------
 */

package config

import "time"

/* ServerConfig is the root configuration structure */
type ServerConfig struct {
	Database DatabaseConfig `json:"database"`
	Server   ServerSettings `json:"server"`
	Logging  LoggingConfig  `json:"logging"`
	Features FeaturesConfig `json:"features"`
	Plugins  []PluginConfig `json:"plugins,omitempty"`
	Middleware []MiddlewareConfig `json:"middleware,omitempty"`
}

/* DatabaseConfig holds database connection configuration */
type DatabaseConfig struct {
	ConnectionString *string   `json:"connectionString,omitempty"`
	Host             *string   `json:"host,omitempty"`
	Port             *int      `json:"port,omitempty"`
	Database         *string   `json:"database,omitempty"`
	User             *string   `json:"user,omitempty"`
	Password         *string   `json:"password,omitempty"`
	Pool             *PoolConfig `json:"pool,omitempty"`
	SSL              interface{} `json:"ssl,omitempty"` // bool or SSLConfig
}

/* PoolConfig holds connection pool settings */
type PoolConfig struct {
	Min                   *int `json:"min,omitempty"`
	Max                   *int `json:"max,omitempty"`
	IdleTimeoutMillis      *int `json:"idleTimeoutMillis,omitempty"`
	ConnectionTimeoutMillis *int `json:"connectionTimeoutMillis,omitempty"`
}

/* SSLConfig holds SSL configuration */
type SSLConfig struct {
	RejectUnauthorized *bool   `json:"rejectUnauthorized,omitempty"`
	CA                 *string `json:"ca,omitempty"`
	Cert               *string `json:"cert,omitempty"`
	Key                *string `json:"key,omitempty"`
}

/* ServerSettings holds server configuration */
type ServerSettings struct {
	Name            *string `json:"name,omitempty"`
	Version         *string `json:"version,omitempty"`
	Timeout         *int    `json:"timeout,omitempty"`
	MaxRequestSize  *int    `json:"maxRequestSize,omitempty"`
	EnableMetrics   *bool   `json:"enableMetrics,omitempty"`
	EnableHealthCheck *bool `json:"enableHealthCheck,omitempty"`
}

/* LoggingConfig holds logging configuration */
type LoggingConfig struct {
	Level              string  `json:"level"`
	Format             string  `json:"format"`
	Output             *string `json:"output,omitempty"`
	EnableRequestLogging  *bool `json:"enableRequestLogging,omitempty"`
	EnableResponseLogging *bool `json:"enableResponseLogging,omitempty"`
	EnableErrorStack      *bool `json:"enableErrorStack,omitempty"`
}

/* FeaturesConfig holds feature flags and settings */
type FeaturesConfig struct {
	Vector        *VectorFeatureConfig        `json:"vector,omitempty"`
	ML            *MLFeatureConfig            `json:"ml,omitempty"`
	Analytics     *AnalyticsFeatureConfig     `json:"analytics,omitempty"`
	RAG           *RAGFeatureConfig          `json:"rag,omitempty"`
	Projects      *ProjectsFeatureConfig      `json:"projects,omitempty"`
	GPU           *GPUFeatureConfig           `json:"gpu,omitempty"`
	Quantization  *QuantizationFeatureConfig  `json:"quantization,omitempty"`
	Dimensionality *DimensionalityFeatureConfig `json:"dimensionality,omitempty"`
	Reranking     *RerankingFeatureConfig    `json:"reranking,omitempty"`
	Hybrid        *HybridFeatureConfig        `json:"hybrid,omitempty"`
	Workers       *WorkersFeatureConfig       `json:"workers,omitempty"`
	Indexing      *IndexingFeatureConfig      `json:"indexing,omitempty"`
}

/* VectorFeatureConfig holds vector feature settings */
type VectorFeatureConfig struct {
	Enabled             bool    `json:"enabled"`
	DefaultDistanceMetric *string `json:"defaultDistanceMetric,omitempty"`
	MaxVectorDimension    *int    `json:"maxVectorDimension,omitempty"`
	DefaultLimit          *int    `json:"defaultLimit,omitempty"`
}

/* MLFeatureConfig holds ML feature settings */
type MLFeatureConfig struct {
	Enabled        bool     `json:"enabled"`
	Algorithms     []string `json:"algorithms,omitempty"`
	MaxTrainingTime *int    `json:"maxTrainingTime,omitempty"`
	GPUEnabled     *bool    `json:"gpuEnabled,omitempty"`
}

/* AnalyticsFeatureConfig holds analytics feature settings */
type AnalyticsFeatureConfig struct {
	Enabled      bool `json:"enabled"`
	MaxClusters  *int `json:"maxClusters,omitempty"`
	MaxIterations *int `json:"maxIterations,omitempty"`
}

/* RAGFeatureConfig holds RAG feature settings */
type RAGFeatureConfig struct {
	Enabled        bool `json:"enabled"`
	DefaultChunkSize *int `json:"defaultChunkSize,omitempty"`
	DefaultOverlap   *int `json:"defaultOverlap,omitempty"`
}

/* ProjectsFeatureConfig holds projects feature settings */
type ProjectsFeatureConfig struct {
	Enabled    bool `json:"enabled"`
	MaxProjects *int `json:"maxProjects,omitempty"`
}

/* GPUFeatureConfig holds GPU feature settings */
type GPUFeatureConfig struct {
	Enabled  bool `json:"enabled"`
	DeviceID *int `json:"deviceId,omitempty"`
}

/* QuantizationFeatureConfig holds quantization feature settings */
type QuantizationFeatureConfig struct {
	Enabled      bool    `json:"enabled"`
	DefaultMethod *string `json:"defaultMethod,omitempty"`
}

/* DimensionalityFeatureConfig holds dimensionality feature settings */
type DimensionalityFeatureConfig struct {
	Enabled      bool `json:"enabled"`
	MaxComponents *int `json:"maxComponents,omitempty"`
}

/* RerankingFeatureConfig holds reranking feature settings */
type RerankingFeatureConfig struct {
	Enabled     bool     `json:"enabled"`
	DefaultModel *string `json:"defaultModel,omitempty"`
}

/* HybridFeatureConfig holds hybrid search feature settings */
type HybridFeatureConfig struct {
	Enabled          bool    `json:"enabled"`
	DefaultVectorWeight *float64 `json:"defaultVectorWeight,omitempty"`
}

/* WorkersFeatureConfig holds workers feature settings */
type WorkersFeatureConfig struct {
	Enabled   bool `json:"enabled"`
	MaxWorkers *int `json:"maxWorkers,omitempty"`
}

/* IndexingFeatureConfig holds indexing feature settings */
type IndexingFeatureConfig struct {
	Enabled              bool `json:"enabled"`
	DefaultHNSWM        *int `json:"defaultHNSWM,omitempty"`
	DefaultHNSWEFConstruction *int `json:"defaultHNSWEFConstruction,omitempty"`
}

/* PluginConfig holds plugin configuration */
type PluginConfig struct {
	Name     string                 `json:"name"`
	Enabled  bool                   `json:"enabled"`
	Path     *string                `json:"path,omitempty"`
	Config   map[string]interface{} `json:"config,omitempty"`
}

/* MiddlewareConfig holds middleware configuration */
type MiddlewareConfig struct {
	Name     string                 `json:"name"`
	Enabled  bool                   `json:"enabled"`
	Order    *int                   `json:"order,omitempty"`
	Config   map[string]interface{} `json:"config,omitempty"`
}

/* Helper methods for getting values with defaults */

func (c *DatabaseConfig) GetHost() string {
	if c.Host != nil {
		return *c.Host
	}
	return "localhost"
}

func (c *DatabaseConfig) GetPort() int {
	if c.Port != nil {
		return *c.Port
	}
	return 5432
}

func (c *DatabaseConfig) GetDatabase() string {
	if c.Database != nil {
		return *c.Database
	}
	return "postgres"
}

func (c *DatabaseConfig) GetUser() string {
	if c.User != nil {
		return *c.User
	}
	return "postgres"
}

func (c *PoolConfig) GetMin() int {
	if c.Min != nil {
		return *c.Min
	}
	return 2
}

func (c *PoolConfig) GetMax() int {
	if c.Max != nil {
		return *c.Max
	}
	return 10
}

func (c *PoolConfig) GetIdleTimeout() time.Duration {
	if c.IdleTimeoutMillis != nil {
		return time.Duration(*c.IdleTimeoutMillis) * time.Millisecond
	}
	return 30 * time.Second
}

func (c *PoolConfig) GetConnectionTimeout() time.Duration {
	if c.ConnectionTimeoutMillis != nil {
		return time.Duration(*c.ConnectionTimeoutMillis) * time.Millisecond
	}
	return 5 * time.Second
}

func (s *ServerSettings) GetName() string {
	if s.Name != nil {
		return *s.Name
	}
	return "neurondb-mcp-server"
}

func (s *ServerSettings) GetVersion() string {
	if s.Version != nil {
		return *s.Version
	}
	return "2.0.0"
}

func (s *ServerSettings) GetTimeout() time.Duration {
	if s.Timeout != nil {
		return time.Duration(*s.Timeout) * time.Millisecond
	}
	return 30 * time.Second
}

