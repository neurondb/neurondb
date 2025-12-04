/*-------------------------------------------------------------------------
 *
 * ml_graph_neural_networks.c
 *    Graph neural network algorithms.
 *
 * This module implements GCN, GAT, and GraphSAGE for graph-structured data
 * learning with model serialization and catalog storage.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_graph_neural_networks.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "neurondb_types.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "neurondb_simd.h"
#include "ml_catalog.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/jsonb.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"

/* GCN layer structure */
typedef struct GCNLayer
{
	float	  **weights;
	float	   *bias;
	int			input_dim;
	int			output_dim;
}			GCNLayer;

/* GCN model structure */
typedef struct GCNModel
{
	GCNLayer   *layers;
	int			n_layers;
	int			hidden_dim;
	int			output_dim;
}			GCNModel;

/*
 * Normalize adjacency matrix (symmetric normalization)
 */
static void
normalize_adjacency(float **adj, int n_nodes, float **norm_adj)
{
	float	   *degrees;
	int			i,
				j;

	degrees = (float *) palloc0(sizeof(float) * n_nodes);

	/* Calculate degrees */
	for (i = 0; i < n_nodes; i++)
		for (j = 0; j < n_nodes; j++)
			degrees[i] += adj[i][j];

	/* Normalize: D^(-1/2) * A * D^(-1/2) */
	for (i = 0; i < n_nodes; i++)
	{
		float		di_sqrt = (degrees[i] > 0.0) ? 1.0 / sqrt(degrees[i]) : 0.0;

		for (j = 0; j < n_nodes; j++)
		{
			float		dj_sqrt = (degrees[j] > 0.0) ? 1.0 / sqrt(degrees[j]) : 0.0;

			norm_adj[i][j] = adj[i][j] * di_sqrt * dj_sqrt;
		}
	}

	NDB_FREE(degrees);
}

/*
 * GCN forward pass: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
 */
static void
gcn_forward(float **features, float **adj_norm, float **weights, float *bias,
			int n_nodes, int input_dim, int output_dim, float **output)
{
	int			i,
				j,
				k;

	/* Matrix multiplication: A_norm * H */
	float	  **ah = (float **) palloc(sizeof(float *) * n_nodes);

	for (i = 0; i < n_nodes; i++)
	{
		ah[i] = (float *) palloc0(sizeof(float) * input_dim);
		for (j = 0; j < n_nodes; j++)
			for (k = 0; k < input_dim; k++)
				ah[i][k] += adj_norm[i][j] * features[j][k];
	}

	/* Matrix multiplication: (A_norm * H) * W */
	for (i = 0; i < n_nodes; i++)
	{
		for (j = 0; j < output_dim; j++)
		{
			output[i][j] = bias[j];
			for (k = 0; k < input_dim; k++)
				output[i][j] += ah[i][k] * weights[k][j];
			/* ReLU activation */
			if (output[i][j] < 0.0)
				output[i][j] = 0.0;
		}
	}

	for (i = 0; i < n_nodes; i++)
		NDB_FREE(ah[i]);
	NDB_FREE(ah);
}

/*
 * gcn_train
 * ---------
 * Train Graph Convolutional Network.
 *
 * SQL Arguments:
 *   graph_table    - Table with graph structure (node_id, neighbor_id, weight)
 *   features_table - Table with node features (node_id, features[])
 *   labels_table   - Table with node labels (node_id, label)
 *   n_nodes        - Number of nodes
 *   feature_dim    - Feature dimension
 *   hidden_dim     - Hidden layer dimension (default: 64)
 *   output_dim     - Output dimension (default: number of classes)
 *   learning_rate  - Learning rate (default: 0.01)
 *   epochs         - Training epochs (default: 100)
 *
 * Returns:
 *   Model ID
 */
PG_FUNCTION_INFO_V1(gcn_train);

Datum
gcn_train(PG_FUNCTION_ARGS)
{
	text	   *graph_table;
	text	   *features_table;
	text	   *labels_table;
	int			n_nodes;
	int			feature_dim;
	int			hidden_dim;
	int			output_dim;
	double		learning_rate;
	int			epochs;
	char	   *graph_tbl;
	char	   *feat_tbl;
	char	   *label_tbl;
	float	  **adjacency;
	float	  **features;
	int		   *labels;
	float	  **adj_norm;
	int			i,
				j;
	int32		model_id;

	graph_table = PG_GETARG_TEXT_PP(0);
	features_table = PG_GETARG_TEXT_PP(1);
	labels_table = PG_GETARG_TEXT_PP(2);
	n_nodes = PG_GETARG_INT32(3);
	feature_dim = PG_GETARG_INT32(4);
	hidden_dim = PG_ARGISNULL(5) ? 64 : PG_GETARG_INT32(5);
	output_dim = PG_ARGISNULL(6) ? 2 : PG_GETARG_INT32(6);
	learning_rate = PG_ARGISNULL(7) ? 0.01 : PG_GETARG_FLOAT8(7);
	epochs = PG_ARGISNULL(8) ? 100 : PG_GETARG_INT32(8);

	if (n_nodes < 1 || feature_dim < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_nodes and feature_dim must be positive")));

	graph_tbl = text_to_cstring(graph_table);
	feat_tbl = text_to_cstring(features_table);
	label_tbl = text_to_cstring(labels_table);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Initialize adjacency matrix */
	NDB_DECLARE(float **, adjacency);
	NDB_DECLARE(float **, adj_norm);
	NDB_ALLOC(adjacency, float *, n_nodes);
	NDB_ALLOC(adj_norm, float *, n_nodes);
	for (i = 0; i < n_nodes; i++)
	{
		NDB_DECLARE(float *, adj_row);
		NDB_DECLARE(float *, adj_norm_row);
		NDB_ALLOC(adj_row, float, n_nodes);
		NDB_ALLOC(adj_norm_row, float, n_nodes);
		adjacency[i] = adj_row;
		adj_norm[i] = adj_norm_row;
		/* Self-connections */
		adjacency[i][i] = 1.0;
	}

	/* Load graph structure */
	{
		StringInfoData query;
		int			ret;

		ndb_spi_stringinfo_init(spi_session, &query);
		appendStringInfo(&query,
						 "SELECT node_id, neighbor_id, COALESCE(weight, 1.0) FROM %s",
						 graph_tbl);

		ret = ndb_spi_execute(spi_session, query.data, true, 0);
		if (ret == SPI_OK_SELECT)
		{
			for (i = 0; i < SPI_processed; i++)
			{
				/* Safe access to SPI_tuptable - validate before access */
				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				{
					continue;
				}
				HeapTuple	tuple = SPI_tuptable->vals[i];
				TupleDesc	tupdesc = SPI_tuptable->tupdesc;
				if (tupdesc == NULL)
				{
					continue;
				}
				/* Use safe function for int32 values */
				int32		node_id_val, neighbor_id_val;
				if (!ndb_spi_get_int32(spi_session, i, 1, &node_id_val))
					continue;
				if (!ndb_spi_get_int32(spi_session, i, 2, &neighbor_id_val))
					continue;
				int			node_id = node_id_val;
				int			neighbor_id = neighbor_id_val;
				/* For float4, need to use SPI_getbinval with safe access */
				float		weight = 0.0f;
				if (tupdesc->natts >= 3)
				{
					Datum		weight_datum = SPI_getbinval(tuple, tupdesc, 3, NULL);
					weight = DatumGetFloat4(weight_datum);
				}

				if (node_id >= 0 && node_id < n_nodes &&
					neighbor_id >= 0 && neighbor_id < n_nodes)
				{
					adjacency[node_id][neighbor_id] = weight;
					adjacency[neighbor_id][node_id] = weight;	/* Undirected */
				}
			}
		}
		ndb_spi_stringinfo_free(spi_session, &query);
	}

	/* Normalize adjacency */
	normalize_adjacency(adjacency, n_nodes, adj_norm);

	/* Load features */
	NDB_DECLARE(float **, features);
	NDB_ALLOC(features, float *, n_nodes);
	for (i = 0; i < n_nodes; i++)
	{
		NDB_DECLARE(float *, feat_row);
		NDB_ALLOC(feat_row, float, feature_dim);
		features[i] = feat_row;
	}

	{
		StringInfoData query;
		int			ret;

		ndb_spi_stringinfo_init(spi_session, &query);
		appendStringInfo(&query,
						 "SELECT node_id, features FROM %s", feat_tbl);

		ret = ndb_spi_execute(spi_session, query.data, true, 0);
		if (ret == SPI_OK_SELECT)
		{
			for (i = 0; i < SPI_processed; i++)
			{
				/* Safe access to SPI_tuptable - validate before access */
				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				{
					continue;
				}
				HeapTuple	tuple = SPI_tuptable->vals[i];
				TupleDesc	tupdesc = SPI_tuptable->tupdesc;
				if (tupdesc == NULL)
				{
					continue;
				}
				/* Use safe function for int32 values */
				int32		node_id_val;
				if (!ndb_spi_get_int32(spi_session, i, 1, &node_id_val))
					continue;
				int			node_id = node_id_val;
				/* For ArrayType, need to use SPI_getbinval with safe access */
				NDB_DECLARE(ArrayType *, feat_array);
				if (tupdesc->natts >= 2)
				{
					Datum		feat_datum = SPI_getbinval(tuple, tupdesc, 2, NULL);
					feat_array = DatumGetArrayTypeP(feat_datum);
				}
				float	   *feat_data;
				int			feat_len;
				int			d;

				if (node_id < 0 || node_id >= n_nodes)
					continue;

				feat_data = (float *) ARR_DATA_PTR(feat_array);
				feat_len = ARR_DIMS(feat_array)[0];

				for (d = 0; d < feature_dim && d < feat_len; d++)
					features[node_id][d] = feat_data[d];
			}
		}
		ndb_spi_stringinfo_free(spi_session, &query);
	}

	/* Load labels */
	NDB_DECLARE(int *, labels);
	NDB_ALLOC(labels, int, n_nodes);
	for (i = 0; i < n_nodes; i++)
		labels[i] = 0;

	{
		StringInfoData query;
		int			ret;

		ndb_spi_stringinfo_init(spi_session, &query);
		appendStringInfo(&query,
						 "SELECT node_id, label FROM %s", label_tbl);

		ret = ndb_spi_execute(spi_session, query.data, true, 0);
		if (ret == SPI_OK_SELECT)
		{
			for (i = 0; i < SPI_processed; i++)
			{
				/* Safe access to SPI_tuptable - validate before access */
				if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
					i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
				{
					continue;
				}
				/* Use safe function to get int32 values */
				int32		node_id_val, label_val;
				if (!ndb_spi_get_int32(spi_session, i, 1, &node_id_val))
					continue;
				if (!ndb_spi_get_int32(spi_session, i, 2, &label_val))
					continue;
				int			node_id = node_id_val;
				int			label = label_val;

				if (node_id >= 0 && node_id < n_nodes)
					labels[node_id] = label;
			}
		}
		ndb_spi_stringinfo_free(spi_session, &query);
	}

	/* Initialize GCN layers */
	GCNLayer	layer1,
				layer2;
	NDB_DECLARE(float **, w1);
	NDB_DECLARE(float **, w2);
	NDB_DECLARE(float *, b1);
	NDB_DECLARE(float *, b2);
	int			l;

	/* Layer 1: feature_dim -> hidden_dim */
	NDB_ALLOC(w1, float *, feature_dim);
	for (i = 0; i < feature_dim; i++)
	{
		NDB_DECLARE(float *, w1_row);
		NDB_ALLOC(w1_row, float, hidden_dim);
		w1[i] = w1_row;
		for (j = 0; j < hidden_dim; j++)
			w1[i][j] = ((float) rand() / (float) RAND_MAX - 0.5) * 0.1;
		}
	NDB_ALLOC(b1, float, hidden_dim);

	/* Layer 2: hidden_dim -> output_dim */
	NDB_ALLOC(w2, float *, hidden_dim);
	for (i = 0; i < hidden_dim; i++)
	{
		NDB_DECLARE(float *, w2_row);
		NDB_ALLOC(w2_row, float, output_dim);
		w2[i] = w2_row;
		for (j = 0; j < output_dim; j++)
			w2[i][j] = ((float) rand() / (float) RAND_MAX - 0.5) * 0.1;
		}
	NDB_ALLOC(b2, float, output_dim);

	/* Training loop (simplified - would need proper backprop) */
	NDB_DECLARE(float **, h1);
	NDB_DECLARE(float **, h2);
	NDB_ALLOC(h1, float *, n_nodes);
	NDB_ALLOC(h2, float *, n_nodes);

	for (i = 0; i < n_nodes; i++)
	{
		NDB_DECLARE(float *, h1_row);
		NDB_DECLARE(float *, h2_row);
		NDB_ALLOC(h1_row, float, hidden_dim);
		NDB_ALLOC(h2_row, float, output_dim);
		h1[i] = h1_row;
		h2[i] = h2_row;
	}

	/* Allocate gradient arrays */
	NDB_DECLARE(float **, grad_h2);
	NDB_DECLARE(float **, grad_h1);
	NDB_DECLARE(float **, grad_w2);
	NDB_DECLARE(float **, grad_w1);
	NDB_DECLARE(float *, grad_b2);
	NDB_DECLARE(float *, grad_b1);
	NDB_ALLOC(grad_h2, float *, n_nodes);
	NDB_ALLOC(grad_h1, float *, n_nodes);
	NDB_ALLOC(grad_w2, float *, hidden_dim);
	NDB_ALLOC(grad_w1, float *, feature_dim);
	NDB_ALLOC(grad_b2, float, output_dim);
	NDB_ALLOC(grad_b1, float, hidden_dim);

	for (i = 0; i < n_nodes; i++)
	{
		NDB_DECLARE(float *, grad_h2_row);
		NDB_DECLARE(float *, grad_h1_row);
		NDB_ALLOC(grad_h2_row, float, output_dim);
		NDB_ALLOC(grad_h1_row, float, hidden_dim);
		grad_h2[i] = grad_h2_row;
		grad_h1[i] = grad_h1_row;
	}

	for (i = 0; i < hidden_dim; i++)
	{
		NDB_DECLARE(float *, grad_w2_row);
		NDB_ALLOC(grad_w2_row, float, output_dim);
		grad_w2[i] = grad_w2_row;
	}
	for (i = 0; i < feature_dim; i++)
	{
		NDB_DECLARE(float *, grad_w1_row);
		NDB_ALLOC(grad_w1_row, float, hidden_dim);
		grad_w1[i] = grad_w1_row;
	}

	for (l = 0; l < epochs; l++)
	{
		float		loss = 0.0f;
		int			node;

		/* Forward pass */
		gcn_forward(features, adj_norm, w1, b1, n_nodes, feature_dim,
					hidden_dim, h1);
		gcn_forward(h1, adj_norm, w2, b2, n_nodes, hidden_dim, output_dim,
					h2);

		/*
		 * Compute cross-entropy loss and output layer gradients using numerically
		 * stable softmax computation. The softmax function exponentiates logits
		 * which can cause numerical overflow when values are large. To prevent
		 * this, we subtract the maximum logit value from all logits before
		 * exponentiation, which shifts the values without changing the softmax
		 * probabilities. This technique ensures that at least one exponentiated
		 * value is exactly 1.0, preventing overflow while maintaining the
		 * mathematical correctness of the softmax distribution. The cross-entropy
		 * loss measures the difference between predicted and true class
		 * distributions, and its gradient with respect to logits is simply the
		 * softmax probabilities minus the one-hot target vector. This gradient
		 * flows backward through the network to update weights during training.
		 */
		for (node = 0; node < n_nodes; node++)
		{
			int			true_label = labels[node];
			float	   *output = h2[node];
			float		sum_exp = 0.0f;
			float		max_logit = -FLT_MAX;
			int			c;

			for (c = 0; c < output_dim; c++)
			{
				if (output[c] > max_logit)
					max_logit = output[c];
			}
			for (c = 0; c < output_dim; c++)
				sum_exp += expf(output[c] - max_logit);

			for (c = 0; c < output_dim; c++)
			{
				float		prob = expf(output[c] - max_logit) / sum_exp;

				if (c == true_label)
				{
					loss -= logf(prob + 1e-10f);
					grad_h2[node][c] = prob - 1.0f; /* dL/dlogit = prob -
													 * target */
				}
				else
				{
					grad_h2[node][c] = prob;	/* dL/dlogit = prob */
				}
			}
		}

		/*
		 * Backpropagate gradients through the second graph convolutional layer.
		 * In graph neural networks, gradients must flow through both the weight
		 * matrix and the normalized adjacency matrix, which aggregates information
		 * from neighboring nodes. The gradient with respect to hidden layer
		 * activations is computed as the transpose of the adjacency matrix
		 * multiplied by output gradients, then multiplied by the transpose of
		 * the weight matrix. This transpose operation reverses the forward
		 * propagation direction, allowing error signals to flow from output nodes
		 * back through the graph structure to hidden layer nodes. The adjacency
		 * matrix transpose ensures that gradients are distributed to neighboring
		 * nodes in proportion to their connection strengths, maintaining the
		 * graph structure's influence on the learning process.
		 */
		{
			float	  **grad_h2_w2 = (float **) palloc(sizeof(float *) * n_nodes);

			for (i = 0; i < n_nodes; i++)
				grad_h2_w2[i] = (float *) palloc0(sizeof(float) * hidden_dim);

			/* grad_h2_w2 = grad_h2 * w2^T */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < hidden_dim; j++)
				{
					for (k = 0; k < output_dim; k++)
						grad_h2_w2[i][j] += grad_h2[i][k] * w2[j][k];
				}
			}

			/* grad_h1 = adj_norm^T * grad_h2_w2 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < n_nodes; j++)
				{
					for (k = 0; k < hidden_dim; k++)
						grad_h1[i][k] += adj_norm[j][i] * grad_h2_w2[j][k];
				}

				/* Apply ReLU derivative */
				for (k = 0; k < hidden_dim; k++)
				{
					if (h1[i][k] <= 0.0f)
						grad_h1[i][k] = 0.0f;
				}
			}

			for (i = 0; i < n_nodes; i++)
				NDB_FREE(grad_h2_w2[i]);
			NDB_FREE(grad_h2_w2);
		}

		/* Compute weight gradients for layer 2 */
		/* grad_w2 = h1^T * adj_norm * grad_h2 */
		{
			float	  **adj_grad_h2 = (float **) palloc(sizeof(float *) * n_nodes);

			for (i = 0; i < n_nodes; i++)
				adj_grad_h2[i] = (float *) palloc0(sizeof(float) * output_dim);

			/* adj_grad_h2 = adj_norm * grad_h2 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < n_nodes; j++)
				{
					for (k = 0; k < output_dim; k++)
						adj_grad_h2[i][k] += adj_norm[i][j] * grad_h2[j][k];
				}
			}

			/* grad_w2 = h1^T * adj_grad_h2 */
			for (i = 0; i < hidden_dim; i++)
			{
				for (j = 0; j < output_dim; j++)
				{
					for (k = 0; k < n_nodes; k++)
						grad_w2[i][j] += h1[k][i] * adj_grad_h2[k][j];
				}
			}

			/* grad_b2 = sum over nodes of grad_h2 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < output_dim; j++)
					grad_b2[j] += grad_h2[i][j];
			}

			for (i = 0; i < n_nodes; i++)
				NDB_FREE(adj_grad_h2[i]);
			NDB_FREE(adj_grad_h2);
		}

		/* Backpropagate through layer 1 */
		/* grad_features = adj_norm^T * grad_h1 * w1^T */
		{
			float	  **grad_h1_w1 = (float **) palloc(sizeof(float *) * n_nodes);

			for (i = 0; i < n_nodes; i++)
				grad_h1_w1[i] = (float *) palloc0(sizeof(float) * feature_dim);

			/* grad_h1_w1 = grad_h1 * w1^T */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < feature_dim; j++)
				{
					for (k = 0; k < hidden_dim; k++)
						grad_h1_w1[i][j] += grad_h1[i][k] * w1[j][k];
				}
			}

			/*
			 * grad_features = adj_norm^T * grad_h1_w1 (not used for weight
			 * update)
			 */
			for (i = 0; i < n_nodes; i++)
				NDB_FREE(grad_h1_w1[i]);
			NDB_FREE(grad_h1_w1);
		}

		/* Compute weight gradients for layer 1 */
		/* grad_w1 = features^T * adj_norm * grad_h1 */
		{
			float	  **adj_grad_h1 = (float **) palloc(sizeof(float *) * n_nodes);

			for (i = 0; i < n_nodes; i++)
				adj_grad_h1[i] = (float *) palloc0(sizeof(float) * hidden_dim);

			/* adj_grad_h1 = adj_norm * grad_h1 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < n_nodes; j++)
				{
					for (k = 0; k < hidden_dim; k++)
						adj_grad_h1[i][k] += adj_norm[i][j] * grad_h1[j][k];
				}
			}

			/* grad_w1 = features^T * adj_grad_h1 */
			for (i = 0; i < feature_dim; i++)
			{
				for (j = 0; j < hidden_dim; j++)
				{
					for (k = 0; k < n_nodes; k++)
						grad_w1[i][j] += features[k][i] * adj_grad_h1[k][j];
				}
			}

			/* grad_b1 = sum over nodes of grad_h1 */
			for (i = 0; i < n_nodes; i++)
			{
				for (j = 0; j < hidden_dim; j++)
					grad_b1[j] += grad_h1[i][j];
			}

			for (i = 0; i < n_nodes; i++)
				NDB_FREE(adj_grad_h1[i]);
			NDB_FREE(adj_grad_h1);
		}

		/* Update weights */
		for (i = 0; i < feature_dim; i++)
		{
			for (j = 0; j < hidden_dim; j++)
				w1[i][j] -= learning_rate * grad_w1[i][j] / (float) n_nodes;
		}
		for (i = 0; i < hidden_dim; i++)
		{
			for (j = 0; j < output_dim; j++)
				w2[i][j] -= learning_rate * grad_w2[i][j] / (float) n_nodes;
		}
		for (i = 0; i < hidden_dim; i++)
			b1[i] -= learning_rate * grad_b1[i] / (float) n_nodes;
		for (i = 0; i < output_dim; i++)
			b2[i] -= learning_rate * grad_b2[i] / (float) n_nodes;

		/* Reset gradients for next iteration */
		for (i = 0; i < n_nodes; i++)
		{
			for (j = 0; j < output_dim; j++)
				grad_h2[i][j] = 0.0f;
			for (j = 0; j < hidden_dim; j++)
				grad_h1[i][j] = 0.0f;
		}
		for (i = 0; i < hidden_dim; i++)
		{
			for (j = 0; j < output_dim; j++)
				grad_w2[i][j] = 0.0f;
		}
		for (i = 0; i < feature_dim; i++)
		{
			for (j = 0; j < hidden_dim; j++)
				grad_w1[i][j] = 0.0f;
		}
		for (i = 0; i < output_dim; i++)
			grad_b2[i] = 0.0f;
		for (i = 0; i < hidden_dim; i++)
			grad_b1[i] = 0.0f;

		if (l % 10 == 0 || l == epochs - 1)
		{
			elog(DEBUG1,
				 "GCN epoch %d: loss = %.6f",
				 l, loss / (float) n_nodes);
		}
	}

	/* Cleanup gradient arrays */
	for (i = 0; i < n_nodes; i++)
	{
		NDB_FREE(grad_h2[i]);
		NDB_FREE(grad_h1[i]);
	}
	for (i = 0; i < hidden_dim; i++)
		NDB_FREE(grad_w2[i]);
	for (i = 0; i < feature_dim; i++)
		NDB_FREE(grad_w1[i]);
	NDB_FREE(grad_h2);
	NDB_FREE(grad_h1);
	NDB_FREE(grad_w2);
	NDB_FREE(grad_w1);
	NDB_FREE(grad_b2);
	NDB_FREE(grad_b1);

	/* Serialize GCN model */
	{
		StringInfoData model_buf;
		NDB_DECLARE(bytea *, serialized);
		StringInfoData paramsbuf;
		StringInfoData metricsbuf;
		NDB_DECLARE(Jsonb *, params_jsonb);
		NDB_DECLARE(Jsonb *, metrics_jsonb);
		MLCatalogModelSpec spec;

		/* Serialize model weights and biases */
		initStringInfo(&model_buf);

		/* Write header */
		pq_sendint32(&model_buf, n_nodes);
		pq_sendint32(&model_buf, feature_dim);
		pq_sendint32(&model_buf, hidden_dim);
		pq_sendint32(&model_buf, output_dim);
		pq_sendint32(&model_buf, n_layers); /* 2 layers */

		/* Serialize layer 1 weights (feature_dim x hidden_dim) */
		for (i = 0; i < feature_dim; i++)
		{
			for (j = 0; j < hidden_dim; j++)
				pq_sendfloat8(&model_buf, w1[i][j]);
		}

		/* Serialize layer 1 bias */
		for (j = 0; j < hidden_dim; j++)
			pq_sendfloat8(&model_buf, b1[j]);

		/* Serialize layer 2 weights (hidden_dim x output_dim) */
		for (i = 0; i < hidden_dim; i++)
		{
			for (j = 0; j < output_dim; j++)
				pq_sendfloat8(&model_buf, w2[i][j]);
		}

		/* Serialize layer 2 bias */
		for (j = 0; j < output_dim; j++)
			pq_sendfloat8(&model_buf, b2[j]);

		serialized = (bytea *) pq_endtypsend(&model_buf);

		/* Build parameters JSON */
		initStringInfo(&paramsbuf);
		appendStringInfo(&paramsbuf,
						 "{\"n_nodes\":%d,"
						 "\"feature_dim\":%d,"
						 "\"hidden_dim\":%d,"
						 "\"output_dim\":%d,"
						 "\"learning_rate\":%.6f,"
						 "\"epochs\":%d,"
						 "\"graph_table\":\"%s\","
						 "\"features_table\":\"%s\","
						 "\"labels_table\":\"%s\"}",
						 n_nodes,
						 feature_dim,
						 hidden_dim,
						 output_dim,
						 learning_rate,
						 epochs,
						 graph_tbl,
						 feat_tbl,
						 label_tbl);
		params_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(paramsbuf.data)));
		NDB_FREE(paramsbuf.data);

		/* Build metrics JSON */
		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
						 "{\"n_nodes\":%d,"
						 "\"training_complete\":true}");
		metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(metricsbuf.data)));
		NDB_FREE(metricsbuf.data);

		/* Register model in catalog */
		memset(&spec, 0, sizeof(MLCatalogModelSpec));
		spec.algorithm = "gcn";
		spec.training_table = graph_tbl;
		spec.model_data = serialized;
		spec.parameters = params_jsonb;
		spec.metrics = metrics_jsonb;

		model_id = ml_catalog_register_model(&spec);
	}

	for (i = 0; i < n_nodes; i++)
	{
		NDB_FREE(adjacency[i]);
		NDB_FREE(adj_norm[i]);
		NDB_FREE(features[i]);
		NDB_FREE(h1[i]);
		NDB_FREE(h2[i]);
	}
	NDB_FREE(adjacency);
	NDB_FREE(adj_norm);
	NDB_FREE(features);
	NDB_FREE(labels);
	for (i = 0; i < feature_dim; i++)
		NDB_FREE(w1[i]);
	NDB_FREE(w1);
	NDB_FREE(b1);
	for (i = 0; i < hidden_dim; i++)
		NDB_FREE(w2[i]);
	NDB_FREE(w2);
	NDB_FREE(b2);
	NDB_FREE(h1);
	NDB_FREE(h2);
	NDB_FREE(graph_tbl);
	NDB_FREE(feat_tbl);
	NDB_FREE(label_tbl);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_INT32(model_id);
}

/*
 * graphsage_aggregate
 * -------------------
 * GraphSAGE neighbor sampling and aggregation.
 *
 * SQL Arguments:
 *   graph_table    - Graph structure table
 *   features_table - Node features table
 *   node_id        - Target node
 *   n_samples      - Number of neighbors to sample (default: 10)
 *   depth          - Aggregation depth (default: 2)
 *
 * Returns:
 *   Aggregated feature vector
 */
PG_FUNCTION_INFO_V1(graphsage_aggregate);

Datum
graphsage_aggregate(PG_FUNCTION_ARGS)
{
	text	   *graph_table;
	text	   *features_table;
	int32		node_id;
	int			n_samples;
	int			depth;
	char	   *graph_tbl;
	char	   *feat_tbl;
	float	   *aggregated;
	int			feature_dim;
	ArrayType  *result;
	Datum	   *result_datums;

	graph_table = PG_GETARG_TEXT_PP(0);
	features_table = PG_GETARG_TEXT_PP(1);
	node_id = PG_GETARG_INT32(2);
	n_samples = PG_ARGISNULL(3) ? 10 : PG_GETARG_INT32(3);
	depth = PG_ARGISNULL(4) ? 2 : PG_GETARG_INT32(4);

	if (n_samples < 1 || depth < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_samples and depth must be positive")));

	graph_tbl = text_to_cstring(graph_table);
	feat_tbl = text_to_cstring(features_table);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Load graph structure and extract feature dimension */
	{
		StringInfoData query;
		int			ret;
		NDB_DECLARE(int *, neighbors);
		int			neighbor_count = 0;
		int			max_neighbors = n_samples * depth;
		NDB_DECLARE(float **, neighbor_features);
		int			sampled_count = 0;

		/* First, get feature dimension from features table */
		ndb_spi_stringinfo_init(spi_session, &query);
		appendStringInfo(&query,
						 "SELECT features FROM %s WHERE node_id = %d LIMIT 1",
						 feat_tbl, node_id);

		ret = ndb_spi_execute(spi_session, query.data, true, 0);
		if (ret == SPI_OK_SELECT && SPI_processed > 0)
		{
			HeapTuple	tuple = SPI_tuptable->vals[0];
			ArrayType  *feat_array = DatumGetArrayTypeP(
														SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, NULL));

			feature_dim = ARR_DIMS(feat_array)[0];
		}
		else
		{
			/* Fallback: try to get dimension from any row */
			ndb_spi_stringinfo_free(spi_session, &query);
			ndb_spi_stringinfo_init(spi_session, &query);
			appendStringInfo(&query,
							 "SELECT features FROM %s LIMIT 1", feat_tbl);
			ret = ndb_spi_execute(spi_session, query.data, true, 0);
			if (ret == SPI_OK_SELECT && SPI_processed > 0)
			{
				HeapTuple	tuple = SPI_tuptable->vals[0];
				ArrayType  *feat_array = DatumGetArrayTypeP(
															SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, NULL));

				feature_dim = ARR_DIMS(feat_array)[0];
			}
			else
			{
				feature_dim = 64;	/* Default fallback */
			}
		}
		ndb_spi_stringinfo_free(spi_session, &query);

		/* Load neighbors from graph structure */
		NDB_ALLOC(neighbors, int, max_neighbors);
		NDB_ALLOC(neighbor_features, float *, max_neighbors);

		/* Sample neighbors at each depth */
		{
			int			current_nodes[1000];	/* Current level nodes */
			int			next_nodes[1000];	/* Next level nodes */
			int			current_count = 1;
			int			next_count = 0;
			int			d;
			int			total_sampled = 0;

			current_nodes[0] = node_id;

			for (d = 0; d < depth && total_sampled < max_neighbors; d++)
			{
				next_count = 0;

				/* For each node at current depth, sample neighbors */
				for (i = 0; i < current_count && total_sampled < max_neighbors; i++)
				{
					int			current_node = current_nodes[i];
					int			samples_needed = (d == depth - 1) ? n_samples : n_samples;
					int			sampled = 0;

					/* Query neighbors from graph table */
					ndb_spi_stringinfo_init(spi_session, &query);
					appendStringInfo(&query,
									 "SELECT neighbor_id FROM %s WHERE node_id = %d ORDER BY random() LIMIT %d",
									 graph_tbl, current_node, samples_needed);

					ret = ndb_spi_execute(spi_session, query.data, true, 0);
					if (ret == SPI_OK_SELECT)
					{
						for (j = 0; j < SPI_processed && sampled < samples_needed && total_sampled < max_neighbors; j++)
						{
							/* Safe access to SPI_tuptable - validate before access */
							if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL || 
								j >= SPI_processed || SPI_tuptable->vals[j] == NULL)
							{
								continue;
							}
							/* Use safe function to get int32 neighbor_id */
							int32		neighbor_id_val;
							if (!ndb_spi_get_int32(spi_session, j, 1, &neighbor_id_val))
								continue;
							int			neighbor_id = neighbor_id_val;

							if (total_sampled < max_neighbors)
							{
								neighbors[total_sampled] = neighbor_id;
								if (next_count < 1000)
									next_nodes[next_count++] = neighbor_id;
								total_sampled++;
								sampled++;
							}
						}
					}
					ndb_spi_stringinfo_free(spi_session, &query);
				}

				/* Move to next depth */
				for (i = 0; i < next_count && i < 1000; i++)
					current_nodes[i] = next_nodes[i];
				current_count = next_count;
			}

			sampled_count = total_sampled;
		}

		/* Load features for sampled neighbors */
		NDB_DECLARE(float *, aggregated);
		NDB_ALLOC(aggregated, float, feature_dim);

		if (sampled_count > 0)
		{
			/* Load features for each neighbor */
			for (i = 0; i < sampled_count; i++)
			{
				ndb_spi_stringinfo_init(spi_session, &query);
				appendStringInfo(&query,
								 "SELECT features FROM %s WHERE node_id = %d",
								 feat_tbl, neighbors[i]);

				ret = ndb_spi_execute(spi_session, query.data, true, 0);
				if (ret == SPI_OK_SELECT && SPI_processed > 0)
				{
					HeapTuple	tuple = SPI_tuptable->vals[0];
					ArrayType  *feat_array = DatumGetArrayTypeP(
																SPI_getbinval(tuple, SPI_tuptable->tupdesc, 1, NULL));
					float	   *feat_data = (float *) ARR_DATA_PTR(feat_array);
					int			feat_len = ARR_DIMS(feat_array)[0];

					NDB_DECLARE(float *, neighbor_feat);
					NDB_ALLOC(neighbor_feat, float, feature_dim);
					neighbor_features[i] = neighbor_feat;
					for (j = 0; j < feature_dim && j < feat_len; j++)
						neighbor_features[i][j] = feat_data[j];
					for (; j < feature_dim; j++)
						neighbor_features[i][j] = 0.0f;
				}
				else
				{
					NDB_DECLARE(float *, neighbor_feat);
					NDB_ALLOC(neighbor_feat, float, feature_dim);
					neighbor_features[i] = neighbor_feat;
				}
				ndb_spi_stringinfo_free(spi_session, &query);
			}

			/* Aggregate: mean of neighbor features */
			for (i = 0; i < sampled_count; i++)
			{
				for (j = 0; j < feature_dim; j++)
					aggregated[j] += neighbor_features[i][j];
			}

			/* Normalize by count */
			if (sampled_count > 0)
			{
				for (j = 0; j < feature_dim; j++)
					aggregated[j] /= sampled_count;
			}

			/* Cleanup neighbor features */
			for (i = 0; i < sampled_count; i++)
				NDB_FREE(neighbor_features[i]);
		}

		NDB_FREE(neighbors);
		NDB_FREE(neighbor_features);
	}

	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * feature_dim);
	for (int i = 0; i < feature_dim; i++)
		result_datums[i] = Float4GetDatum(aggregated[i]);

	result = construct_array(result_datums,
							 feature_dim,
							 FLOAT4OID,
							 sizeof(float4),
							 FLOAT4PASSBYVAL,
							 'i');

	NDB_FREE(aggregated);
	NDB_FREE(result_datums);
	NDB_FREE(graph_tbl);
	NDB_FREE(feat_tbl);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_ARRAYTYPE_P(result);
}
