/*-------------------------------------------------------------------------
 *
 * prometheus.c
 *      Prometheus-compatible HTTP metrics exporter (detailed implementation)
 *
 * This module provides complete, thread-safe collection and serving
 * of NeurondB metrics for consumption by Prometheus.
 *
 * It exposes detailed database metrics via an HTTP endpoint, which can
 * be scraped by Prometheus to provide real-time cluster observability.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *        src/metrics/prometheus.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "neurondb_scan.h"
#include "fmgr.h"
#include "access/htup_details.h"
#include "libpq/pqsignal.h"
#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "postmaster/interrupt.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/typcache.h"
#include "funcapi.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_macros.h"

/* HTTP server configuration (GUC-backed settings) */
static int	prometheus_port = 9187;
static bool prometheus_enabled = false;
static char *prometheus_host = NULL;

/* Prometheus metric names (should be stable and documented) */
#define METRIC_QUERIES_TOTAL "neurondb_queries_total"
#define METRIC_QUERY_DURATION "neurondb_query_duration_seconds"
#define METRIC_INDEX_SIZE "neurondb_index_size_bytes"
#define METRIC_VECTOR_COUNT "neurondb_vector_count"
#define METRIC_CACHE_HITS "neurondb_cache_hits_total"
#define METRIC_CACHE_MISSES "neurondb_cache_misses_total"
#define METRIC_WORKER_STATUS "neurondb_worker_status"

/*
 * PrometheusMetrics: Detailed shared-memory structure to store
 * all relevant metrics, with one protecting LWLock pointer.
 * All accesses (read or write) must be under proper LWLock.
 */
typedef struct PrometheusMetrics
{
	LWLock	   *lock;			/* Pointer to lightweight lock */

	/* Query-related metrics */
	int64		queries_total;	/* Total queries run */
	int64		queries_success;	/* Successful queries */
	int64		queries_error;	/* Failed queries */
	float8		query_duration_sum; /* Sum of query durations in seconds */
	float8		query_duration_max; /* Max query duration seen */

	/* Index-related metrics */
	int64		vectors_total;	/* Total vectors indexed */
	int64		index_size_bytes;	/* Current raw index size (bytes) */
	int64		index_inserts;	/* Index inserts (for future use) */
	int64		index_deletes;	/* Index deletes (for future use) */

	/* Cache metrics */
	int64		cache_hits;		/* Total cache hits */
	int64		cache_misses;	/* Total cache misses */
	int64		cache_evictions;	/* Total evictions (unused here) */

	/* Worker pool/run metrics */
	int			workers_active; /* Current active workers */
	int			workers_idle;	/* Current idle workers */
	int64		jobs_processed; /* Jobs processed (for future use) */
	int64		jobs_failed;	/* Jobs failed (for future use) */

}			PrometheusMetrics;

static PrometheusMetrics * prom_metrics = NULL;

/* Forward declarations */
static void prometheus_worker_main(Datum arg);
static void handle_http_request(int client_socket);
static void send_metrics(int socket);
static Size prometheus_shmem_size(void);
static void prometheus_shmem_init(void);

/*
 * prometheus_init_guc
 * Registers the custom parameters (GUCs) that control prometheus exporter.
 * Requires postmaster restart to change.
 */
void
prometheus_init_guc(void)
{
	DefineCustomIntVariable("neurondb.prometheus_port",
							"Port for Prometheus metrics HTTP endpoint",
							NULL,
							&prometheus_port,
							9187,
							1024,
							65535,
							PGC_POSTMASTER,
							0,
							NULL,
							NULL,
							NULL);

	DefineCustomBoolVariable("neurondb.prometheus_enabled",
							 "Enable Prometheus metrics exporter",
							 NULL,
							 &prometheus_enabled,
							 false,
							 PGC_POSTMASTER,
							 0,
							 NULL,
							 NULL,
							 NULL);

	DefineCustomStringVariable("neurondb.prometheus_host",
							   "Host address to bind Prometheus endpoint",
							   NULL,
							   &prometheus_host,
							   "0.0.0.0",
							   PGC_POSTMASTER,
							   0,
							   NULL,
							   NULL,
							   NULL);
}

/*
 * prometheus_register_worker
 * Registers the exporter as a background worker; actual work done in main loop.
 */
void
prometheus_register_worker(void)
{
	BackgroundWorker worker;

	if (!prometheus_enabled)
		return;

	memset(&worker, 0, sizeof(BackgroundWorker));
	strcpy(worker.bgw_name, "neurondb prometheus exporter");
	strcpy(worker.bgw_type, "neurondb prometheus");
	worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
	worker.bgw_start_time = BgWorkerStart_PostmasterStart;
	worker.bgw_restart_time = 10;	/* restart after 10 seconds on crash */
	sprintf(worker.bgw_library_name, "neurondb");
	sprintf(worker.bgw_function_name, "prometheus_worker_main");
	worker.bgw_main_arg = (Datum) 0;
	worker.bgw_notify_pid = 0;

	RegisterBackgroundWorker(&worker);

	elog(LOG,
		 "neurondb: Registered Prometheus exporter worker on port %d",
		 prometheus_port);
}

/*
 * prometheus_shmem_size
 * Returns requested size for shared-memory allocator
 */
static Size
prometheus_shmem_size(void)
{
	return MAXALIGN(sizeof(PrometheusMetrics));
}

/*
 * prometheus_shmem_init
 * Allocates and initializes metric counters, LWLock, only at postmaster startup.
 */
static void
prometheus_shmem_init(void)
{
	bool		found = false;

	prom_metrics = (PrometheusMetrics *) ShmemInitStruct(
														 "neurondb_prometheus_metrics", prometheus_shmem_size(), &found);

	if (!found)
	{
		/* Single LWLock per metrics struct, grabbed from our named tranche */
		prom_metrics->lock =
			&(GetNamedLWLockTranche("neurondb_prometheus"))->lock;
		/* Zero all stats fields */
		prom_metrics->queries_total = 0;
		prom_metrics->queries_success = 0;
		prom_metrics->queries_error = 0;
		prom_metrics->query_duration_sum = 0.0;
		prom_metrics->query_duration_max = 0.0;
		prom_metrics->vectors_total = 0;
		prom_metrics->index_size_bytes = 0;
		prom_metrics->index_inserts = 0;
		prom_metrics->index_deletes = 0;
		prom_metrics->cache_hits = 0;
		prom_metrics->cache_misses = 0;
		prom_metrics->cache_evictions = 0;
		prom_metrics->workers_active = 0;
		prom_metrics->workers_idle = 0;
		prom_metrics->jobs_processed = 0;
		prom_metrics->jobs_failed = 0;
	}
}

/*
 * prometheus_worker_main
 * The HTTP server--runs in its own background worker process, loops and accepts connections.
 */
__attribute__((unused)) static void
prometheus_worker_main(Datum arg)
{
	int			listen_socket = -1;
	int			client_socket = -1;
	struct sockaddr_in server_addr;
	struct sockaddr_in client_addr;
	socklen_t	client_len = 0;
	int			opt = 1;

	/* Set up signal handlers to allow safe shutdown */
	pqsignal(SIGTERM, SignalHandlerForShutdownRequest);
	pqsignal(SIGINT, SignalHandlerForShutdownRequest);
	BackgroundWorkerUnblockSignals();

	/* Attach to shared memory and initialize if needed */
	prometheus_shmem_init();

	elog(LOG,
		 "neurondb: Prometheus exporter starting on %s:%d",
		 prometheus_host,
		 prometheus_port);

	/* Create TCP listening socket */
	listen_socket = socket(AF_INET, SOCK_STREAM, 0);
	if (listen_socket < 0)
	{
		ereport(ERROR,
				(errmsg("neurondb: Failed to create Prometheus socket: "
						"%m")));
	}

	/* Allow re-binding (helpful for quick restarts) */
	if (setsockopt(
				   listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))
		< 0)
	{
		close(listen_socket);
		ereport(ERROR,
				(errmsg("neurondb: setsockopt(SO_REUSEADDR) failed: "
						"%m")));
	}

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = inet_addr(prometheus_host);
	server_addr.sin_port = htons(prometheus_port);

	/* Bind socket to specified address/port */
	if (bind(listen_socket,
			 (struct sockaddr *) &server_addr,
			 sizeof(server_addr))
		< 0)
	{
		close(listen_socket);
		ereport(ERROR,
				(errmsg("neurondb: Failed to bind Prometheus socket to "
						"%s:%d: %m",
						prometheus_host,
						prometheus_port)));
	}

	/* Listen for incoming TCP connections, backlog = 5 */
	if (listen(listen_socket, 5) < 0)
	{
		close(listen_socket);
		ereport(ERROR,
				(errmsg("neurondb: Failed to listen on Prometheus "
						"socket: %m")));
	}

	/* The HTTP accept/serve loop */
	while (!ShutdownRequestPending)
	{
		client_len = sizeof(client_addr);
		errno = 0;
		client_socket = accept(listen_socket,
							   (struct sockaddr *) &client_addr,
							   &client_len);

		if (client_socket < 0)
		{
			if (errno == EINTR)
				continue;
			continue;
		}

		/* For each accepted connection, serve a single HTTP request */
		handle_http_request(client_socket);
		close(client_socket);

		/* Check for interrupts or signals (postmaster/shutdown, etc) */
		CHECK_FOR_INTERRUPTS();
	}

	close(listen_socket);
	proc_exit(0);
}

/*
 * handle_http_request
 * Parses the first read request, serves either /metrics or simple HTML index or 404.
 */
static void
handle_http_request(int client_socket)
{
	char		buffer[4096];
	ssize_t		bytes_read = 0;

	/* Read client request. For simplicity, only handle single line. */
	bytes_read = read(client_socket, buffer, sizeof(buffer) - 1);
	if (bytes_read < 0)
		return;

	buffer[bytes_read] = '\0';

	/* Strict comparison allows both HTTP/1.0 and 1.1 simple GETs */
	if (strncmp(buffer, "GET /metrics", 12) == 0)
	{
		send_metrics(client_socket);
	}
	else if (strncmp(buffer, "GET / ", 6) == 0
			 || strncmp(buffer, "GET / HTTP", 10) == 0)
	{
		/* Serve a tiny HTML page at GET / */
		const char *response = "HTTP/1.1 200 OK\r\n"
			"Content-Type: text/html\r\n\r\n"
			"<html><body>"
			"<h1>NeurondB Prometheus Exporter</h1>"
			"<p>Metrics available at <a "
			"href=\"/metrics\">/metrics</a></p>"
			"</body></html>";

		{
			ssize_t __w = write(client_socket, response, strlen(response));
			(void) __w;
		}
	}
	else
	{
		/* Anything else gets a 404 Not Found */
		const char *response = "HTTP/1.1 404 Not Found\r\n"
			"Content-Type: text/plain\r\n\r\n"
			"Not Found";

		{
			ssize_t __w = write(client_socket, response, strlen(response));
			(void) __w;
		}
	}
}

/*
 * send_metrics
 * Actually formats the in-memory counters as Prometheus exposition format with HELP/TYPE lines.
 * Acquires the metrics lock in shared mode for thread-safe reads.
 */
static void
send_metrics(int socket)
{
	StringInfoData metrics;

	initStringInfo(&metrics);

	/* Basic HTTP/Prometheus exposition header */
	appendStringInfo(&metrics,
					 "HTTP/1.1 200 OK\r\n"
					 "Content-Type: text/plain; version=0.0.4\r\n\r\n");

	/* Acquire shared lock on metrics data while reading */
	LWLockAcquire(prom_metrics->lock, LW_SHARED);

	/* ----- Metrics output ----- */

	/* Total queries--counter */
	appendStringInfo(&metrics,
					 "# HELP %s Total number of queries (all types, successes and "
					 "errors)\n"
					 "# TYPE %s counter\n"
					 "%s %lld\n\n",
					 METRIC_QUERIES_TOTAL,
					 METRIC_QUERIES_TOTAL,
					 METRIC_QUERIES_TOTAL,
					 (long long) prom_metrics->queries_total);

	/* Query duration--summary (sum + max, for now) */
	appendStringInfo(&metrics,
					 "# HELP %s Aggregate duration of all queries in seconds\n"
					 "# TYPE %s summary\n"
					 "%s_sum %.6f\n"
					 "%s_max %.6f\n\n",
					 METRIC_QUERY_DURATION,
					 METRIC_QUERY_DURATION,
					 METRIC_QUERY_DURATION,
					 prom_metrics->query_duration_sum,
					 METRIC_QUERY_DURATION,
					 prom_metrics->query_duration_max);

	/* Index: vectors count--gauge */
	appendStringInfo(&metrics,
					 "# HELP %s Number of vectors currently indexed\n"
					 "# TYPE %s gauge\n"
					 "%s %lld\n\n",
					 METRIC_VECTOR_COUNT,
					 METRIC_VECTOR_COUNT,
					 METRIC_VECTOR_COUNT,
					 (long long) prom_metrics->vectors_total);

	/* Index: storage size in bytes--gauge */
	appendStringInfo(&metrics,
					 "# HELP %s Logical size of the NeurondB index in bytes\n"
					 "# TYPE %s gauge\n"
					 "%s %lld\n\n",
					 METRIC_INDEX_SIZE,
					 METRIC_INDEX_SIZE,
					 METRIC_INDEX_SIZE,
					 (long long) prom_metrics->index_size_bytes);

	/* Cache hits--counter */
	appendStringInfo(&metrics,
					 "# HELP %s Number of successful cache lookups\n"
					 "# TYPE %s counter\n"
					 "%s %lld\n\n",
					 METRIC_CACHE_HITS,
					 METRIC_CACHE_HITS,
					 METRIC_CACHE_HITS,
					 (long long) prom_metrics->cache_hits);

	/* Cache misses--counter */
	appendStringInfo(&metrics,
					 "# HELP %s Number of failed cache lookups\n"
					 "# TYPE %s counter\n"
					 "%s %lld\n\n",
					 METRIC_CACHE_MISSES,
					 METRIC_CACHE_MISSES,
					 METRIC_CACHE_MISSES,
					 (long long) prom_metrics->cache_misses);

	/* Worker status--gauge (multi-value label) */
	appendStringInfo(&metrics,
					 "# HELP %s Current NeurondB worker status\n"
					 "# TYPE %s gauge\n"
					 "%s{status=\"active\"} %d\n"
					 "%s{status=\"idle\"} %d\n\n",
					 METRIC_WORKER_STATUS,
					 METRIC_WORKER_STATUS,
					 METRIC_WORKER_STATUS,
					 prom_metrics->workers_active,
					 METRIC_WORKER_STATUS,
					 prom_metrics->workers_idle);

	/* Release shared metrics lock */
	LWLockRelease(prom_metrics->lock);

	/*
	 * Send HTTP/Prometheus response to socket as a single buffer. (No chunked
	 * encoding, blocking I/O)
	 */
	{
		ssize_t __w = write(socket, metrics.data, metrics.len);
		(void) __w;
	}
	NDB_FREE(metrics.data);
}

/*
 * Public update APIs, convenient for use within NeurondB's other code,
 * each method does lock-protected increments and max/accumulator logic as needed.
 */
void
prometheus_record_query(float8 duration_seconds, bool success)
{
	if (prom_metrics == NULL)
		return;

	LWLockAcquire(prom_metrics->lock, LW_EXCLUSIVE);

	prom_metrics->queries_total++;
	if (success)
		prom_metrics->queries_success++;
	else
		prom_metrics->queries_error++;

	prom_metrics->query_duration_sum += duration_seconds;
	if (duration_seconds > prom_metrics->query_duration_max)
		prom_metrics->query_duration_max = duration_seconds;

	LWLockRelease(prom_metrics->lock);
}

void
prometheus_record_cache_hit(void)
{
	if (prom_metrics == NULL)
		return;
	LWLockAcquire(prom_metrics->lock, LW_EXCLUSIVE);
	prom_metrics->cache_hits++;
	LWLockRelease(prom_metrics->lock);
}

void
prometheus_record_cache_miss(void)
{
	if (prom_metrics == NULL)
		return;
	LWLockAcquire(prom_metrics->lock, LW_EXCLUSIVE);
	prom_metrics->cache_misses++;
	LWLockRelease(prom_metrics->lock);
}

/*
 * SQL-callable function to get a tuple of major Prometheus metrics.
 * This can be SELECTed from SQL, returning:
 *   (queries_total, queries_success, queries_error, query_duration_sum,
 *    vectors_total, cache_hits, cache_misses, workers_active)
 */
PG_FUNCTION_INFO_V1(neurondb_prometheus_metrics);

Datum
neurondb_prometheus_metrics(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[8];
	bool		nulls[8];
	HeapTuple	tuple;

	if (prom_metrics == NULL)
		ereport(ERROR,
				(errmsg("neurondb: Prometheus metrics not "
						"initialized")));

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("function returning record called in "
						"context that cannot accept type "
						"record")));

	tupdesc = BlessTupleDesc(tupdesc);

	LWLockAcquire(prom_metrics->lock, LW_SHARED);

	values[0] = Int64GetDatum(prom_metrics->queries_total);
	values[1] = Int64GetDatum(prom_metrics->queries_success);
	values[2] = Int64GetDatum(prom_metrics->queries_error);
	values[3] = Float8GetDatum(prom_metrics->query_duration_sum);
	values[4] = Int64GetDatum(prom_metrics->vectors_total);
	values[5] = Int64GetDatum(prom_metrics->cache_hits);
	values[6] = Int64GetDatum(prom_metrics->cache_misses);
	values[7] = Int32GetDatum(prom_metrics->workers_active);

	LWLockRelease(prom_metrics->lock);

	memset(nulls, 0, sizeof(nulls));

	tuple = heap_form_tuple(tupdesc, values, nulls);

	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}
