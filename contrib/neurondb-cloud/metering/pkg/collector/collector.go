package collector

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

const defaultWindowMinutes = 60

// Config holds metering collector configuration.
type Config struct {
	DatabaseURL        string
	WindowMinutes      int
	DryRun             bool
	TenantMetricsURLFn func(tenantID, orgID string) string // optional: URL to fetch metrics per tenant
}

// Record is a single metering record to be written.
type Record struct {
	OrgID       string
	TenantID    string
	MetricType  string
	Quantity    float64
	WindowStart time.Time
	WindowEnd   time.Time
	Metadata    map[string]interface{}
}

// Collector runs the metering collection cycle.
type Collector struct {
	cfg   Config
	pool  *pgxpool.Pool
	log   *slog.Logger
}

// New creates a new Collector. Caller must call Close() when done.
func New(ctx context.Context, cfg Config) (*Collector, error) {
	if cfg.WindowMinutes <= 0 {
		cfg.WindowMinutes = defaultWindowMinutes
	}
	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return nil, fmt.Errorf("database: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("database ping: %w", err)
	}
	log := slog.Default().With("component", "metering-collector")
	return &Collector{cfg: cfg, pool: pool, log: log}, nil
}

// Close closes the database pool.
func (c *Collector) Close() {
	c.pool.Close()
}

// Run performs one collection cycle: list tenants, gather usage, write records.
func (c *Collector) Run(ctx context.Context) (int, error) {
	windowEnd := time.Now().UTC().Truncate(time.Minute)
	windowStart := windowEnd.Add(-time.Duration(c.cfg.WindowMinutes) * time.Minute)

	tenants, err := c.listTenants(ctx)
	if err != nil {
		return 0, fmt.Errorf("list tenants: %w", err)
	}

	var records []Record
	for _, t := range tenants {
		recs, err := c.gatherForTenant(ctx, t.orgID, t.tenantID, windowStart, windowEnd)
		if err != nil {
			c.log.Warn("gather tenant failed", "tenant_id", t.tenantID, "error", err)
			continue
		}
		records = append(records, recs...)
	}

	if c.cfg.DryRun {
		c.log.Info("dry run", "records", len(records), "window_start", windowStart, "window_end", windowEnd)
		return len(records), nil
	}

	written, err := c.insertRecords(ctx, records)
	if err != nil {
		return 0, fmt.Errorf("insert: %w", err)
	}
	c.log.Info("cycle complete", "written", written, "window_start", windowStart, "window_end", windowEnd)
	return written, nil
}

type tenantRow struct {
	orgID    string
	tenantID string
}

func (c *Collector) listTenants(ctx context.Context) ([]tenantRow, error) {
	// Control plane schema: tenants table with org_id and id. Adapt if your schema uses different columns.
	rows, err := c.pool.Query(ctx,
		`SELECT COALESCE(org_id::text, '00000000-0000-0000-0000-000000000000'), id::text FROM tenants LIMIT 1000`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []tenantRow
	for rows.Next() {
		var t tenantRow
		if err := rows.Scan(&t.orgID, &t.tenantID); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

func (c *Collector) gatherForTenant(ctx context.Context, orgID, tenantID string, start, end time.Time) ([]Record, error) {
	if c.cfg.TenantMetricsURLFn != nil {
		url := c.cfg.TenantMetricsURLFn(tenantID, orgID)
		if url != "" {
			recs, err := c.fetchMetricsFromURL(ctx, url, orgID, tenantID, start, end)
			if err != nil {
				c.log.Warn("fetch metrics failed", "tenant_id", tenantID, "url", url, "error", err)
			} else if len(recs) > 0 {
				return recs, nil
			}
		}
	}
	/* Heartbeat record per tenant when no metrics URL or fetch returned nothing */
	return []Record{{
		OrgID:       orgID,
		TenantID:    tenantID,
		MetricType:  "api_calls",
		Quantity:    0,
		WindowStart: start,
		WindowEnd:   end,
		Metadata:    map[string]interface{}{"source": "metering-collector", "type": "heartbeat"},
	}}, nil
}

/* fetchMetricsFromURL GETs url and parses JSON body: { "metric_name": number, ... } into one Record per metric */
func (c *Collector) fetchMetricsFromURL(ctx context.Context, url, orgID, tenantID string, start, end time.Time) ([]Record, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics endpoint returned %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 512*1024))
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, fmt.Errorf("parse metrics JSON: %w", err)
	}
	var records []Record
	for k, v := range raw {
		if v == nil {
			continue
		}
		var qty float64
		switch x := v.(type) {
		case float64:
			qty = x
		case int:
			qty = float64(x)
		case int64:
			qty = float64(x)
		case string:
			qty, _ = strconv.ParseFloat(strings.TrimSpace(x), 64)
		default:
			continue
		}
		metricType := strings.ReplaceAll(k, " ", "_")
		if metricType == "" {
			metricType = "metric"
		}
		records = append(records, Record{
			OrgID:       orgID,
			TenantID:    tenantID,
			MetricType:  metricType,
			Quantity:    qty,
			WindowStart: start,
			WindowEnd:   end,
			Metadata:    map[string]interface{}{"source": "metrics_endpoint", "url": url},
		})
	}
	return records, nil
}

func (c *Collector) insertRecords(ctx context.Context, records []Record) (int, error) {
	metaJSON := func(m map[string]interface{}) []byte {
		if m == nil {
			return []byte("{}")
		}
		b, _ := json.Marshal(m)
		return b
	}
	var n int
	for _, r := range records {
		_, err := c.pool.Exec(ctx,
			`INSERT INTO metering_records (org_id, tenant_id, metric_type, quantity, window_start, window_end, metadata)
			 VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7::jsonb)`,
			r.OrgID, r.TenantID, r.MetricType, r.Quantity, r.WindowStart, r.WindowEnd, metaJSON(r.Metadata),
		)
		if err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}
