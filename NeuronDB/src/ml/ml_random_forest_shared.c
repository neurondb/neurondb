/*-------------------------------------------------------------------------
 *
 * ml_random_forest_shared.c
 *    Random Forest implementation for classification and regression
 *
 * Implements Random Forest as an ensemble of decision trees using
 * bootstrap aggregating (bagging) and random feature selection.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_random_forest_shared.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "lib/stringinfo.h"
#include "utils/builtins.h"

#include "gtree.h"
#include "ml_random_forest_shared.h"
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

/*
 * rf_gini_impurity
 *    Compute Gini impurity for a set of class label counts.
 *
 * Gini impurity measures the probability of misclassifying a randomly selected
 * element if it were labeled according to the class distribution in the set.
 * The impurity is computed as one minus the sum of squared class probabilities,
 * where each probability is the ratio of class count to total count. When all
 * elements belong to the same class, the impurity is zero, indicating perfect
 * purity. When elements are evenly distributed across classes, the impurity
 * reaches its maximum value, indicating maximum uncertainty. The function
 * iterates through all class counts, computes the squared probability for each
 * class, sums these squared probabilities, and subtracts the sum from one to
 * obtain the final impurity value. Lower impurity indicates better class
 * separation and more informative splits, making this metric suitable for
 * evaluating split quality in decision tree construction algorithms.
 */
double
rf_gini_impurity(const int *counts, int n_classes, int total)
{
	double		sum_sq = 0.0;
	int			i;

	if (counts == NULL || n_classes <= 0 || total <= 0)
		return 0.0;

	for (i = 0; i < n_classes; i++)
	{
		if (counts[i] > 0)
		{
			double		p = (double) counts[i] / (double) total;

			sum_sq += p * p;
		}
	}
	return 1.0 - sum_sq;
}

/* Split Gini impurity (left/right counts) */
double
rf_split_gini(const int *left_counts,
			  const int *right_counts,
			  int class_count,
			  int *left_total_out,
			  int *right_total_out,
			  int *left_majority_out,
			  int *right_majority_out)
{
	int			left_total = 0;
	int			right_total = 0;
	int			left_majority = 0;
	int			right_majority = 0;
	int			left_best = 0;
	int			right_best = 0;
	double		gini = 0.0;
	int			i;

	if (class_count <= 0)
	{
		if (left_total_out)
			*left_total_out = 0;
		if (right_total_out)
			*right_total_out = 0;
		if (left_majority_out)
			*left_majority_out = 0;
		if (right_majority_out)
			*right_majority_out = 0;
		return DBL_MAX;
	}

	/* Defensive check: ensure we don't access invalid memory */
	if ((left_counts == NULL && right_counts == NULL) ||
		(left_counts != NULL && right_counts != NULL))
	{
		/* Both NULL or both non-NULL is acceptable */
	}
	else
	{
		/* Mixed NULL/non-NULL is unusual but handle gracefully */
	}

	for (i = 0; i < class_count; i++)
	{
		int			l = left_counts ? left_counts[i] : 0;
		int			r = right_counts ? right_counts[i] : 0;

		left_total += l;
		right_total += r;

		if (l > left_best)
		{
			left_best = l;
			left_majority = i;
		}
		if (r > right_best)
		{
			right_best = r;
			right_majority = i;
		}
	}

	if (left_total <= 0 && right_total <= 0)
	{
		if (left_total_out)
			*left_total_out = 0;
		if (right_total_out)
			*right_total_out = 0;
		if (left_majority_out)
			*left_majority_out = 0;
		if (right_majority_out)
			*right_majority_out = 0;
		return DBL_MAX;
	}

	if (left_total > 0)
	{
		double		inv_total = 1.0 / (double) left_total;
		double		purity = 0.0;

		for (i = 0; i < class_count; i++)
		{
			double		p = (double) (left_counts ? left_counts[i] : 0)
				* inv_total;

			purity += p * p;
		}
		gini += ((double) left_total
				 / (double) (left_total + right_total))
			* (1.0 - purity);
	}

	if (right_total > 0)
	{
		double		inv_total = 1.0 / (double) right_total;
		double		purity = 0.0;

		for (i = 0; i < class_count; i++)
		{
			double		p = (double) (right_counts ? right_counts[i] : 0)
				* inv_total;

			purity += p * p;
		}
		gini += ((double) right_total
				 / (double) (left_total + right_total))
			* (1.0 - purity);
	}

	if (left_total_out)
		*left_total_out = left_total;
	if (right_total_out)
		*right_total_out = right_total;
	if (left_majority_out)
		*left_majority_out = left_majority;
	if (right_majority_out)
		*right_majority_out = right_majority;

	return gini;
}

void
rf_tree_iterate_nodes(const GTree *tree, rf_node_iter_fn iter, void *arg)
{
	const GTreeNode *nodes;
	int			i;

	if (tree == NULL || iter == NULL)
		return;

	nodes = gtree_nodes(tree);
	if (nodes == NULL)
		return;

	for (i = 0; i < tree->count; i++)
		iter(arg, &nodes[i], i);
}

Jsonb *
rf_build_metrics_json(const RFMetricsSpec *spec)
{
	StringInfoData buf;
	bool		first = true;
	Jsonb	   *result;
	Datum		jsonb_datum;

	elog(DEBUG1, "neurondb: rf_build_metrics_json: function entry");

	if (spec == NULL)
	{
		elog(DEBUG1, "neurondb: rf_build_metrics_json: spec is NULL, returning NULL");
		return NULL;
	}

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: about to initStringInfo")));

	initStringInfo(&buf);

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: StringInfo initialized, about to append opening brace")));

	appendStringInfoChar(&buf, '{');

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: opening brace appended")));

	if (spec->storage != NULL)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: appending storage field"),
				 errdetail("storage=%s", spec->storage)));
		appendStringInfo(&buf, "\"storage\":\"%s\"", spec->storage);
		first = false;
	}

	if (spec->algorithm != NULL)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: appending algorithm field"),
				 errdetail("algorithm=%s, first=%s", spec->algorithm, first ? "true" : "false")));
		appendStringInfo(&buf,
						 "%s\"algorithm\":\"%s\"",
						 first ? "" : ",",
						 spec->algorithm);
		first = false;
	}

	if (spec->tree_count >= 0)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: appending tree_count field"),
				 errdetail("tree_count=%d, first=%s", spec->tree_count, first ? "true" : "false")));
		appendStringInfo(&buf,
						 "%s\"n_trees\":%d",
						 first ? "" : ",",
						 spec->tree_count);
		first = false;
	}

	if (spec->majority_class >= 0)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: appending majority_class field"),
				 errdetail("majority_class=%d, first=%s", spec->majority_class, first ? "true" : "false")));
		appendStringInfo(&buf,
						 "%s\"majority_class\":%d",
						 first ? "" : ",",
						 spec->majority_class);
		first = false;
	}

	if (spec->majority_fraction >= 0.0)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: appending majority_fraction field"),
				 errdetail("majority_fraction=%f, first=%s", spec->majority_fraction, first ? "true" : "false")));
		appendStringInfo(&buf,
						 "%s\"majority_fraction\":%.6f",
						 first ? "" : ",",
						 spec->majority_fraction);
		first = false;
	}

	if (spec->gini >= 0.0)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: appending gini field"),
				 errdetail("gini=%f, first=%s", spec->gini, first ? "true" : "false")));
		appendStringInfo(
						 &buf, "%s\"gini\":%.6f", first ? "" : ",", spec->gini);
		first = false;
	}

	if (spec->oob_accuracy >= 0.0)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: appending oob_accuracy field"),
				 errdetail("oob_accuracy=%f, first=%s", spec->oob_accuracy, first ? "true" : "false")));
		appendStringInfo(&buf,
						 "%s\"oob_accuracy\":%.6f",
						 first ? "" : ",",
						 spec->oob_accuracy);
		first = false;
	}

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: about to append closing brace"),
			 errdetail("buf.data=%p, buf.len=%d", (void *) buf.data, buf.len)));

	appendStringInfoChar(&buf, '}');

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: closing brace appended"),
			 errdetail("buf.data=%p, buf.len=%d", (void *) buf.data, buf.len)));

	if (buf.data == NULL)
	{
		elog(DEBUG1, "neurondb: rf_build_metrics_json: buf.data is NULL, returning NULL");
		return NULL;
	}

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: about to call DirectFunctionCall1(jsonb_in)"),
			 errdetail("buf.data=%s, buf.len=%d, CurrentMemoryContext=%p",
					   buf.data ? buf.data : "NULL", buf.len, (void *) CurrentMemoryContext)));

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: about to call CStringGetTextDatum"),
			 errdetail("buf.data=%p", (void *) buf.data)));

	jsonb_datum = CStringGetTextDatum(buf.data);

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: CStringGetTextDatum returned"),
			 errdetail("jsonb_datum=%lu", (unsigned long) jsonb_datum)));

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: about to call DirectFunctionCall1"),
			 errdetail("jsonb_datum=%lu", (unsigned long) jsonb_datum)));

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, jsonb_datum));

	ereport(DEBUG2,
			(errmsg("rf_build_metrics_json: DirectFunctionCall1 returned"),
			 errdetail("result=%p", (void *) result)));

	if (buf.data != NULL)
	{
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: about to free buf.data")));
		NDB_FREE(buf.data);
		ereport(DEBUG2,
				(errmsg("rf_build_metrics_json: buf.data freed")));
	}

	elog(DEBUG1, "neurondb: rf_build_metrics_json: about to return result");

	return result;
}
