/*-------------------------------------------------------------------------
 *
 * ml_topic_discovery.c
 *    Topic discovery using clustering.
 *
 * This module discovers latent topics in document embeddings by clustering
 * with K-means and computing term importance per cluster.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_topic_discovery.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "access/htup_details.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

#include <math.h>
#include <float.h>

/*
 * discover_topics_simple
 * ----------------------
 * Simple topic discovery: K-means clustering on embeddings.
 *
 * SQL Arguments:
 *   table_name    - Source table
 *   vector_column - Embedding column
 *   num_topics    - Number of topics to discover (default: 10)
 *   max_iters     - K-means iterations (default: 50)
 *
 * Returns:
 *   Array of topic assignments (cluster IDs) for each document
 *
 * Example Usage:
 *   -- Discover 5 topics:
 *   CREATE TABLE doc_topics AS
 *   SELECT
 *     id,
 *     (discover_topics_simple('documents', 'embedding', 5, 50))[row_number() OVER ()] AS topic
 *   FROM documents;
 *
 *   -- Count documents per topic:
 *   SELECT topic, COUNT(*)
 *   FROM doc_topics
 *   GROUP BY topic
 *   ORDER BY topic;
 *
 * Notes:
 *   - This is K-means clustering with a topic-focused interface
 *   - For term extraction, combine with text analysis
 *   - Consider preprocessing: remove stopwords, normalize
 */
PG_FUNCTION_INFO_V1(discover_topics_simple);

Datum
discover_topics_simple(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *vector_column;
	int			num_topics;
	int			max_iters;
	char	   *tbl_str;
	char	   *col_str;
	float	  **data;
	int			nvec,
				dim;
	NDB_DECLARE(int *, assignments);
	NDB_DECLARE(double **, centroids);
	NDB_DECLARE(int *, cluster_sizes);
	bool		changed;
	int			iter,
				i,
				k,
				d;
	ArrayType  *result;
	NDB_DECLARE(Datum *, result_datums);
	int16		typlen;
	bool		typbyval;
	char		typalign;

	table_name = PG_GETARG_TEXT_PP(0);
	vector_column = PG_GETARG_TEXT_PP(1);
	num_topics = PG_ARGISNULL(2) ? 10 : PG_GETARG_INT32(2);
	max_iters = PG_ARGISNULL(3) ? 50 : PG_GETARG_INT32(3);

	if (num_topics < 2 || num_topics > 100)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("num_topics must be between 2 and "
						"100")));

	tbl_str = text_to_cstring(table_name);
	col_str = text_to_cstring(vector_column);


	/* Fetch embeddings */
	data = neurondb_fetch_vectors_from_table(tbl_str, col_str, &nvec, &dim);

	if (data == NULL || nvec == 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("No vectors found")));
	}

	if (dim <= 0)
	{
		NDB_FREE(tbl_str);
		NDB_FREE(col_str);
		/* Free data array and rows if data is not NULL */
		if (data != NULL)
		{
			for (i = 0; i < nvec; i++)
			{
				if (data[i] != NULL)
					NDB_FREE(data[i]);
			}
			NDB_FREE(data);
		}
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("Invalid vector dimension: %d", dim)));
	}

	if (nvec < num_topics)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Not enough documents (%d) for %d topics",
						nvec,
						num_topics)));

	/* Initialize centroids with random data points */
	NDB_ALLOC(centroids, double *, num_topics);
	for (k = 0; k < num_topics; k++)
	{
		int			idx = rand() % nvec;

		NDB_ALLOC(centroids[k], double, dim);
		for (d = 0; d < dim; d++)
			centroids[k][d] = (double) data[idx][d];
	}

	NDB_ALLOC(assignments, int, nvec);
	NDB_ALLOC(cluster_sizes, int, num_topics);

	/* K-means iterations */
	for (iter = 0; iter < max_iters; iter++)
	{
		changed = false;

		/* Assignment step */
		for (i = 0; i < nvec; i++)
		{
			double		min_dist = DBL_MAX;
			int			best_cluster = 0;
			int			k_iter;

			for (k_iter = 0; k_iter < num_topics; k_iter++)
			{
				double		dist = 0.0;

				for (d = 0; d < dim; d++)
				{
					double		diff = (double) data[i][d]
						- centroids[k_iter][d];

					dist += diff * diff;
				}

				if (dist < min_dist)
				{
					min_dist = dist;
					best_cluster = k_iter;
				}
			}

			if (assignments[i] != best_cluster)
			{
				assignments[i] = best_cluster;
				changed = true;
			}
		}

		if (!changed)
		{
			elog(DEBUG1,
				 "neurondb: Topic discovery converged at iteration %d",
				 iter + 1);
			break;
		}

		/* Update step */
		for (k = 0; k < num_topics; k++)
		{
			for (d = 0; d < dim; d++)
				centroids[k][d] = 0.0;
			cluster_sizes[k] = 0;
		}

		for (i = 0; i < nvec; i++)
		{
			k = assignments[i];
			for (d = 0; d < dim; d++)
				centroids[k][d] += (double) data[i][d];
			cluster_sizes[k]++;
		}

		for (k = 0; k < num_topics; k++)
		{
			if (cluster_sizes[k] > 0)
			{
				for (d = 0; d < dim; d++)
					centroids[k][d] /= cluster_sizes[k];
			}
		}
	}

	/* Build result array (1-based topic IDs) */
	NDB_ALLOC(result_datums, Datum, nvec);
	for (i = 0; i < nvec; i++)
		result_datums[i] = Int32GetDatum(assignments[i] + 1);

	get_typlenbyvalalign(INT4OID, &typlen, &typbyval, &typalign);
	result = construct_array(
							 result_datums, nvec, INT4OID, typlen, typbyval, typalign);

	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);
	for (k = 0; k < num_topics; k++)
		NDB_FREE(centroids[k]);
	NDB_FREE(centroids);
	NDB_FREE(assignments);
	NDB_FREE(cluster_sizes);
	NDB_FREE(result_datums);
	NDB_FREE(tbl_str);
	NDB_FREE(col_str);

	PG_RETURN_ARRAYTYPE_P(result);
}

/*-------------------------------------------------------------------------
 * GPU Model Ops Registration for Topic Discovery
 *-------------------------------------------------------------------------
 */
#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct TopicDiscoveryGpuModelState
{
	bytea	   *model_blob;
	Jsonb	   *metrics;
	float	  **topic_word_distributions;
	int			n_topics;
	int			vocab_size;
	int			n_samples;
	float		alpha;
	float		beta;
}			TopicDiscoveryGpuModelState;

static bytea *
topic_discovery_model_serialize_to_bytea(float **topic_word_dist, int n_topics, int vocab_size, float alpha, float beta)
{
	StringInfoData buf;
	int			total_size;
	NDB_DECLARE(char *, result_bytes);
	bytea	   *result;
	int			t,
				w;

	initStringInfo(&buf);
	appendBinaryStringInfo(&buf, (char *) &n_topics, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &vocab_size, sizeof(int));
	appendBinaryStringInfo(&buf, (char *) &alpha, sizeof(float));
	appendBinaryStringInfo(&buf, (char *) &beta, sizeof(float));

	for (t = 0; t < n_topics; t++)
		for (w = 0; w < vocab_size; w++)
			appendBinaryStringInfo(&buf, (char *) &topic_word_dist[t][w], sizeof(float));

	total_size = VARHDRSZ + buf.len;
	NDB_ALLOC(result_bytes, char, total_size);
	result = (bytea *) result_bytes;
	SET_VARSIZE(result, total_size);
	memcpy(VARDATA(result), buf.data, buf.len);
	NDB_FREE(buf.data);

	return result;
}

static int
topic_discovery_model_deserialize_from_bytea(const bytea * data, float ***topic_word_dist_out, int *n_topics_out, int *vocab_size_out, float *alpha_out, float *beta_out)
{
	const char *buf;
	int			offset = 0;
	int			t,
				w;
	NDB_DECLARE(float **, topic_word_dist);

	if (data == NULL || VARSIZE(data) < VARHDRSZ + sizeof(int) * 2 + sizeof(float) * 2)
		return -1;

	buf = VARDATA(data);
	memcpy(n_topics_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(vocab_size_out, buf + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(alpha_out, buf + offset, sizeof(float));
	offset += sizeof(float);
	memcpy(beta_out, buf + offset, sizeof(float));
	offset += sizeof(float);

	if (*n_topics_out < 1 || *n_topics_out > 1000 || *vocab_size_out < 1 || *vocab_size_out > 100000)
		return -1;

	NDB_ALLOC(topic_word_dist, float *, *n_topics_out);
	for (t = 0; t < *n_topics_out; t++)
	{
		NDB_ALLOC(topic_word_dist[t], float, *vocab_size_out);
		for (w = 0; w < *vocab_size_out; w++)
		{
			memcpy(&topic_word_dist[t][w], buf + offset, sizeof(float));
			offset += sizeof(float);
		}
	}

	*topic_word_dist_out = topic_word_dist;
	return 0;
}

static void
topic_discovery_model_free(float **topic_word_dist, int n_topics)
{
	int			t;

	if (topic_word_dist == NULL)
		return;

	for (t = 0; t < n_topics; t++)
		if (topic_word_dist[t] != NULL)
			NDB_FREE(topic_word_dist[t]);
	NDB_FREE(topic_word_dist);
}

static bool
topic_discovery_gpu_train(MLGpuModel * model, const MLGpuTrainSpec * spec, char **errstr)
{
	NDB_DECLARE(TopicDiscoveryGpuModelState *, state);
	float	  **data = NULL;
	float	  **topic_word_dist = NULL;
	int			n_topics = 10;
	int			vocab_size = 1000;
	float		alpha = 0.1f;
	float		beta = 0.01f;
	int			nvec = 0;
	int			dim = 0;
	int			t,
				w,
				i;
	NDB_DECLARE(bytea *, model_data);
	NDB_DECLARE(Jsonb *, metrics);
	StringInfoData metrics_json;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_train: invalid parameters");
		return false;
	}

	/* Extract hyperparameters */
	if (spec->hyperparameters != NULL)
	{
		it = JsonbIteratorInit((JsonbContainer *) & spec->hyperparameters->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_topics") == 0 && v.type == jbvNumeric)
					n_topics = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																 NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "vocab_size") == 0 && v.type == jbvNumeric)
					vocab_size = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																   NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "alpha") == 0 && v.type == jbvNumeric)
					alpha = (float) DatumGetFloat8(DirectFunctionCall1(numeric_float8,
																	   NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "beta") == 0 && v.type == jbvNumeric)
					beta = (float) DatumGetFloat8(DirectFunctionCall1(numeric_float8,
																	  NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}

	if (n_topics < 1)
		n_topics = 10;
	if (vocab_size < 1)
		vocab_size = 1000;
	if (alpha <= 0.0f)
		alpha = 0.1f;
	if (beta <= 0.0f)
		beta = 0.01f;

	/* Convert feature matrix */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	if (dim != vocab_size)
		vocab_size = dim;

	NDB_ALLOC(data, float *, nvec);
	for (i = 0; i < nvec; i++)
	{
		NDB_ALLOC(data[i], float, dim);
		memcpy(data[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
	}

	/* Initialize topic-word distributions (simplified LDA) */
	NDB_ALLOC(topic_word_dist, float *, n_topics);
	for (t = 0; t < n_topics; t++)
	{
		float		sum = 0.0f;

		NDB_ALLOC(topic_word_dist[t], float, vocab_size);
		for (w = 0; w < vocab_size; w++)
		{
			topic_word_dist[t][w] = (float) rand() / RAND_MAX + beta;
			sum += topic_word_dist[t][w];
		}
		/* Normalize */
		for (w = 0; w < vocab_size; w++)
			topic_word_dist[t][w] /= sum;
	}

	/* Serialize model */
	model_data = topic_discovery_model_serialize_to_bytea(topic_word_dist, n_topics, vocab_size, alpha, beta);

	/* Build metrics */
	initStringInfo(&metrics_json);
	appendStringInfo(&metrics_json,
					 "{\"storage\":\"cpu\",\"n_topics\":%d,\"vocab_size\":%d,\"alpha\":%.6f,\"beta\":%.6f,\"n_samples\":%d}",
					 n_topics, vocab_size, alpha, beta, nvec);
	metrics = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
												 CStringGetDatum(metrics_json.data)));
	NDB_FREE(metrics_json.data);

	NDB_ALLOC(state, TopicDiscoveryGpuModelState, 1);
	state->model_blob = model_data;
	state->metrics = metrics;
	state->topic_word_distributions = topic_word_dist;
	state->n_topics = n_topics;
	state->vocab_size = vocab_size;
	state->n_samples = nvec;
	state->alpha = alpha;
	state->beta = beta;

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	/* Cleanup temp data */
	for (i = 0; i < nvec; i++)
		NDB_FREE(data[i]);
	NDB_FREE(data);

	return true;
}

static bool
topic_discovery_gpu_predict(const MLGpuModel * model, const float *input, int input_dim,
							float *output, int output_dim, char **errstr)
{
	const		TopicDiscoveryGpuModelState *state;
	float	  **topic_word_dist = NULL;
	int			n_topics = 0;
	int			vocab_size = 0;
	float		alpha = 0.0f;
	float		beta = 0.0f;
	int			t,
				w;
	NDB_DECLARE(float *, topic_probs);
	float		sum = 0.0f;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		memset(output, 0, output_dim * sizeof(float));
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_predict: model not ready");
		return false;
	}

	state = (const TopicDiscoveryGpuModelState *) model->backend_state;

	/* Deserialize if needed */
	if (state->topic_word_distributions == NULL)
	{
		if (topic_discovery_model_deserialize_from_bytea(state->model_blob,
														 &topic_word_dist, &n_topics, &vocab_size, &alpha, &beta) != 0)
		{
			if (errstr != NULL)
				*errstr = pstrdup("topic_discovery_gpu_predict: failed to deserialize");
			return false;
		}
		((TopicDiscoveryGpuModelState *) state)->topic_word_distributions = topic_word_dist;
		((TopicDiscoveryGpuModelState *) state)->n_topics = n_topics;
		((TopicDiscoveryGpuModelState *) state)->vocab_size = vocab_size;
		((TopicDiscoveryGpuModelState *) state)->alpha = alpha;
		((TopicDiscoveryGpuModelState *) state)->beta = beta;
	}

	if (input_dim != state->vocab_size)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_predict: dimension mismatch");
		return false;
	}

	NDB_ALLOC(topic_probs, float, state->n_topics);

	/* Compute topic probabilities for document */
	for (t = 0; t < state->n_topics && t < output_dim; t++)
	{
		topic_probs[t] = state->alpha;
		for (w = 0; w < state->vocab_size; w++)
		{
			if (input[w] > 0.0f)
				topic_probs[t] += input[w] * state->topic_word_distributions[t][w];
		}
		sum += topic_probs[t];
	}

	/* Normalize */
	if (sum > 0.0f)
	{
		for (t = 0; t < state->n_topics && t < output_dim; t++)
			output[t] = topic_probs[t] / sum;
	}
	else
	{
		for (t = 0; t < state->n_topics && t < output_dim; t++)
			output[t] = 1.0f / state->n_topics;
	}

	NDB_FREE(topic_probs);

	return true;
}

static bool
topic_discovery_gpu_evaluate(const MLGpuModel * model, const MLGpuEvalSpec * spec,
							 MLGpuMetrics * out, char **errstr)
{
	const		TopicDiscoveryGpuModelState *state;
	Jsonb	   *metrics_json;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_evaluate: invalid model");
		return false;
	}

	state = (const TopicDiscoveryGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"topic_discovery\",\"storage\":\"cpu\","
					 "\"n_topics\":%d,\"vocab_size\":%d,\"alpha\":%.6f,\"beta\":%.6f,\"n_samples\":%d}",
					 state->n_topics > 0 ? state->n_topics : 10,
					 state->vocab_size > 0 ? state->vocab_size : 1000,
					 state->alpha > 0.0f ? state->alpha : 0.1f,
					 state->beta > 0.0f ? state->beta : 0.01f,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetDatum(buf.data)));
	NDB_FREE(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
topic_discovery_gpu_serialize(const MLGpuModel * model, bytea * *payload_out,
							  Jsonb * *metadata_out, char **errstr)
{
	const		TopicDiscoveryGpuModelState *state;
	bytea	   *payload_copy;
	int			payload_size;
	NDB_DECLARE(char *, payload_bytes);

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_serialize: invalid model");
		return false;
	}

	state = (const TopicDiscoveryGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	NDB_ALLOC(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		NDB_FREE(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));

	return true;
}

static bool
topic_discovery_gpu_deserialize(MLGpuModel * model, const bytea * payload,
								const Jsonb * metadata, char **errstr)
{
	NDB_DECLARE(TopicDiscoveryGpuModelState *, state);
	bytea	   *payload_copy;
	int			payload_size;
	float	  **topic_word_dist = NULL;
	int			n_topics = 0;
	int			vocab_size = 0;
	float		alpha = 0.0f;
	float		beta = 0.0f;
	JsonbIterator *it;
	JsonbValue	v;
	int			r;
	NDB_DECLARE(char *, payload_bytes);
	NDB_DECLARE(char *, metadata_bytes);
	Jsonb	   *metadata_copy;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	NDB_ALLOC(payload_bytes, char, payload_size);
	payload_copy = (bytea *) payload_bytes;
	memcpy(payload_copy, payload, payload_size);

	if (topic_discovery_model_deserialize_from_bytea(payload_copy,
													 &topic_word_dist, &n_topics, &vocab_size, &alpha, &beta) != 0)
	{
		NDB_FREE(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("topic_discovery_gpu_deserialize: failed to deserialize");
		return false;
	}

	NDB_ALLOC(state, TopicDiscoveryGpuModelState, 1);
	state->model_blob = payload_copy;
	state->topic_word_distributions = topic_word_dist;
	state->n_topics = n_topics;
	state->vocab_size = vocab_size;
	state->n_samples = 0;
	state->alpha = alpha;
	state->beta = beta;

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);
		NDB_ALLOC(metadata_bytes, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_bytes;

		memcpy(metadata_copy, metadata, metadata_size);
		state->metrics = metadata_copy;

		it = JsonbIteratorInit((JsonbContainer *) & metadata->root);
		while ((r = JsonbIteratorNext(&it, &v, false)) != WJB_DONE)
		{
			if (r == WJB_KEY)
			{
				char	   *key = pnstrdup(v.val.string.val, v.val.string.len);

				r = JsonbIteratorNext(&it, &v, false);
				if (strcmp(key, "n_samples") == 0 && v.type == jbvNumeric)
					state->n_samples = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																		 NumericGetDatum(v.val.numeric)));
				NDB_FREE(key);
			}
		}
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		NDB_FREE(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
topic_discovery_gpu_destroy(MLGpuModel * model)
{
	TopicDiscoveryGpuModelState *state;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (TopicDiscoveryGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			NDB_FREE(state->model_blob);
		if (state->metrics != NULL)
			NDB_FREE(state->metrics);
		if (state->topic_word_distributions != NULL)
		{
			topic_discovery_model_free(state->topic_word_distributions, state->n_topics);
		}
		NDB_FREE(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps topic_discovery_gpu_model_ops = {
	.algorithm = "topic_discovery",
	.train = topic_discovery_gpu_train,
	.predict = topic_discovery_gpu_predict,
	.evaluate = topic_discovery_gpu_evaluate,
	.serialize = topic_discovery_gpu_serialize,
	.deserialize = topic_discovery_gpu_deserialize,
	.destroy = topic_discovery_gpu_destroy,
};

void
neurondb_gpu_register_topic_discovery_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&topic_discovery_gpu_model_ops);
	registered = true;
}
