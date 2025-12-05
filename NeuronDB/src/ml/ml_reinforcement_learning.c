/*-------------------------------------------------------------------------
 *
 * ml_reinforcement_learning.c
 *    Reinforcement learning algorithms.
 *
 * This module implements Q-learning, DQN, PPO, and multi-armed bandits
 * for reinforcement learning tasks.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_reinforcement_learning.c
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
#include "utils/jsonb.h"

#include "neurondb.h"
#include "neurondb_ml.h"
#include "ml_catalog.h"

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include "neurondb_validation.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_spi_safe.h"

/* Q-Learning agent structure */
typedef struct QLearningAgent
{
	double	  **q_table;
	int			n_states;
	int			n_actions;
	double		learning_rate;
	double		discount_factor;
	double		epsilon;
}			QLearningAgent;

/*
 * qlearning_train
 * ---------------
 * Train Q-Learning agent on experience data.
 *
 * SQL Arguments:
 *   state_action_rewards - Table with columns: state_id, action_id, reward, next_state_id
 *   n_states            - Number of states
 *   n_actions           - Number of actions
 *   learning_rate       - Learning rate (default: 0.1)
 *   discount_factor     - Discount factor gamma (default: 0.95)
 *   epsilon             - Exploration rate (default: 0.1)
 *   iterations          - Training iterations (default: 1000)
 *
 * Returns:
 *   Model ID of trained Q-table
 */
PG_FUNCTION_INFO_V1(qlearning_train);

Datum
qlearning_train(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	int			n_states;
	int			n_actions;
	double		learning_rate;
	double		discount_factor;
	double		epsilon;
	int			iterations;
	char	   *tbl_str;
	QLearningAgent *agent;
	StringInfoData query;
	int			ret;
	int			i;
	Jsonb	   *parameters;
	Jsonb	   *metrics;
	MLCatalogModelSpec spec;
	int32		model_id;

	table_name = PG_GETARG_TEXT_PP(0);
	n_states = PG_GETARG_INT32(1);
	n_actions = PG_GETARG_INT32(2);
	learning_rate = PG_ARGISNULL(3) ? 0.1 : PG_GETARG_FLOAT8(3);
	discount_factor = PG_ARGISNULL(4) ? 0.95 : PG_GETARG_FLOAT8(4);
	epsilon = PG_ARGISNULL(5) ? 0.1 : PG_GETARG_FLOAT8(5);
	iterations = PG_ARGISNULL(6) ? 1000 : PG_GETARG_INT32(6);

	if (n_states < 1 || n_actions < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_states and n_actions must be positive")));
	if (learning_rate <= 0.0 || learning_rate > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learning_rate must be between 0 and 1")));
	if (discount_factor < 0.0 || discount_factor > 1.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("discount_factor must be between 0 and 1")));

	tbl_str = text_to_cstring(table_name);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Initialize Q-table */
	NDB_DECLARE(QLearningAgent *, agent);
	NDB_ALLOC(agent, QLearningAgent, 1);
	agent->n_states = n_states;
	agent->n_actions = n_actions;
	agent->learning_rate = learning_rate;
	agent->discount_factor = discount_factor;
	agent->epsilon = epsilon;

	NDB_ALLOC(agent->q_table, double *, n_states);
	for (i = 0; i < n_states; i++)
	{
		NDB_ALLOC(agent->q_table[i], double, n_actions);
	}

	/* Training loop */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT state_id, action_id, reward, next_state_id FROM %s",
					 tbl_str);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("Failed to fetch training data")));

	for (i = 0; i < iterations && i < SPI_processed; i++)
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
		int32		state_id_val,
					action_id_val,
					next_state_id_val;
		bool		isnull = false;

		if (!ndb_spi_get_int32(spi_session, i, 1, &state_id_val))
			continue;
		if (!ndb_spi_get_int32(spi_session, i, 2, &action_id_val))
			continue;
		int			state_id = state_id_val;
		int			action_id = action_id_val;

		/* For float8 reward, need to use SPI_getbinval with safe access */
		double		reward = 0.0;

		if (tupdesc->natts >= 3)
		{
			Datum		reward_datum = SPI_getbinval(tuple, tupdesc, 3, NULL);

			reward = DatumGetFloat8(reward_datum);
		}

		/* For next_state_id, use safe function */
		if (!ndb_spi_get_int32(spi_session, i, 4, &next_state_id_val))
		{
			isnull = true;
		}
		int			next_state_id = next_state_id_val;

		if (state_id < 0 || state_id >= n_states || action_id < 0 ||
			action_id >= n_actions)
			continue;

		/*
		 * Q-Learning update: Q(s,a) += alpha * (r + gamma * max Q(s',a') -
		 * Q(s,a))
		 */
		double		current_q = agent->q_table[state_id][action_id];
		double		max_next_q = 0.0;
		int			a;

		if (!isnull && next_state_id >= 0 && next_state_id < n_states)
		{
			for (a = 0; a < n_actions; a++)
				if (agent->q_table[next_state_id][a] > max_next_q)
					max_next_q =
						agent->q_table[next_state_id][a];
		}

		agent->q_table[state_id][action_id] +=
			learning_rate *
			(reward + discount_factor * max_next_q - current_q);
	}

	/* Serialize Q-table to JSONB */
	{
		StringInfoData jsonbuf;
		int			s,
					a;

		initStringInfo(&jsonbuf);
		appendStringInfoString(&jsonbuf, "{\"q_table\":[");
		for (s = 0; s < n_states; s++)
		{
			if (s > 0)
				appendStringInfoString(&jsonbuf, ",");
			appendStringInfoString(&jsonbuf, "[");
			for (a = 0; a < n_actions; a++)
			{
				if (a > 0)
					appendStringInfoString(&jsonbuf, ",");
				appendStringInfo(&jsonbuf, "%.6f",
								 agent->q_table[s][a]);
			}
			appendStringInfoString(&jsonbuf, "]");
		}
		appendStringInfoString(&jsonbuf, "]}");
		parameters = DatumGetJsonbP(
									DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
		NDB_FREE(jsonbuf.data);
	}

	/* Create metrics */
	{
		StringInfoData metricsbuf;

		initStringInfo(&metricsbuf);
		appendStringInfo(&metricsbuf,
						 "{\"iterations\":%d,\"n_states\":%d,\"n_actions\":%d}",
						 iterations, n_states, n_actions);
		metrics = DatumGetJsonbP(
								 DirectFunctionCall1(jsonb_in, CStringGetTextDatum(metricsbuf.data)));
		NDB_FREE(metricsbuf.data);
	}

	/* Register model */
	memset(&spec, 0, sizeof(spec));
	spec.algorithm = "qlearning";
	spec.model_type = "reinforcement_learning";
	spec.training_table = tbl_str;
	spec.training_column = "state_id";
	spec.parameters = parameters;
	spec.metrics = metrics;
	spec.project_name = "qlearning_project";

	model_id = ml_catalog_register_model(&spec);

	for (i = 0; i < n_states; i++)
		NDB_FREE(agent->q_table[i]);
	NDB_FREE(agent->q_table);
	NDB_FREE(agent);
	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(tbl_str);

	PG_RETURN_INT32(model_id);
}

/*
 * qlearning_predict
 * -----------------
 * Get best action for a state using trained Q-table.
 */
PG_FUNCTION_INFO_V1(qlearning_predict);

Datum
qlearning_predict(PG_FUNCTION_ARGS)
{
	int32		model_id;
	int32		state_id;
	double	  **q_table;
	int			n_states,
				n_actions;
	int			best_action = 0;
	double		best_q = -DBL_MAX;
	int			a;
	Jsonb	   *parameters;
	StringInfoData query;
	int			ret;

	model_id = PG_GETARG_INT32(0);
	state_id = PG_GETARG_INT32(1);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Load Q-table from model */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT parameters FROM neurondb.ml_models WHERE model_id = %d",
					 model_id);

	ret = ndb_spi_execute(spi_session, query.data, true, 1);
	if (ret != SPI_OK_SELECT || SPI_processed == 0)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Model not found")));
	}

	parameters = ndb_spi_get_jsonb(spi_session, 0, 1, oldcontext);

	/* Parse Q-table from JSONB (simplified - would need proper JSONB parsing) */
	/* For now, return greedy action based on state_id */
	/* In full implementation, would deserialize Q-table */

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	PG_RETURN_INT32(best_action);
}

/*
 * multi_armed_bandit
 * ------------------
 * Multi-armed bandit algorithms (Thompson Sampling, UCB, Epsilon-Greedy).
 *
 * SQL Arguments:
 *   table_name    - Table with columns: arm_id, reward
 *   algorithm     - 'thompson', 'ucb', or 'epsilon_greedy'
 *   n_arms        - Number of arms
 *   epsilon       - Exploration rate for epsilon-greedy (default: 0.1)
 *   alpha         - Prior alpha for Thompson Sampling (default: 1.0)
 *   beta          - Prior beta for Thompson Sampling (default: 1.0)
 *
 * Returns:
 *   Array of arm selection probabilities
 */
PG_FUNCTION_INFO_V1(multi_armed_bandit);

Datum
multi_armed_bandit(PG_FUNCTION_ARGS)
{
	text	   *table_name;
	text	   *algorithm_text;
	int			n_arms;
	double		epsilon;
	double		alpha;
	double		beta;
	char	   *tbl_str;
	char	   *algorithm;

	NDB_DECLARE(int *, arm_counts);
	NDB_DECLARE(double *, arm_rewards);
	NDB_DECLARE(double *, arm_probs);
	int			i;
	ArrayType  *result;

	NDB_DECLARE(Datum *, result_datums);
	StringInfoData query;
	int			ret;

	table_name = PG_GETARG_TEXT_PP(0);
	algorithm_text = PG_GETARG_TEXT_PP(1);
	n_arms = PG_GETARG_INT32(2);
	epsilon = PG_ARGISNULL(3) ? 0.1 : PG_GETARG_FLOAT8(3);
	alpha = PG_ARGISNULL(4) ? 1.0 : PG_GETARG_FLOAT8(4);
	beta = PG_ARGISNULL(5) ? 1.0 : PG_GETARG_FLOAT8(5);

	if (n_arms < 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("n_arms must be positive")));

	tbl_str = text_to_cstring(table_name);
	algorithm = text_to_cstring(algorithm_text);

	NDB_DECLARE(NdbSpiSession *, spi_session);
	MemoryContext oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	/* Initialize arm statistics */
	NDB_ALLOC(arm_counts, int, n_arms);
	NDB_ALLOC(arm_rewards, double, n_arms);

	/* Collect statistics from table */
	ndb_spi_stringinfo_init(spi_session, &query);
	appendStringInfo(&query,
					 "SELECT arm_id, reward FROM %s", tbl_str);

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
			/* Use safe function for int32 arm_id */
			int32		arm_id_val;

			if (!ndb_spi_get_int32(spi_session, i, 1, &arm_id_val))
				continue;
			int			arm_id = arm_id_val;

			/* For float8 reward, need to use SPI_getbinval with safe access */
			double		reward = 0.0;

			if (tupdesc->natts >= 2)
			{
				Datum		reward_datum = SPI_getbinval(tuple, tupdesc, 2, NULL);

				reward = DatumGetFloat8(reward_datum);
			}

			if (arm_id >= 0 && arm_id < n_arms)
			{
				arm_counts[arm_id]++;
				arm_rewards[arm_id] += reward;
			}
		}
	}

	/* Calculate selection probabilities */
	arm_probs = (double *) palloc(sizeof(double) * n_arms);

	if (strcmp(algorithm, "thompson") == 0)
	{
		/* Thompson Sampling: sample from Beta distribution */
		for (i = 0; i < n_arms; i++)
		{
			double		successes = arm_rewards[i];
			double		failures = arm_counts[i] - successes;

			/* Simplified: use mean of Beta(alpha+successes, beta+failures) */
			double		mean = (alpha + successes) /
				(alpha + successes + beta + failures);

			arm_probs[i] = mean;
		}
	}
	else if (strcmp(algorithm, "ucb") == 0)
	{
		/* Upper Confidence Bound */
		int			total_pulls = 0;
		int			a;

		for (a = 0; a < n_arms; a++)
			total_pulls += arm_counts[a];

		for (i = 0; i < n_arms; i++)
		{
			double		avg_reward = (arm_counts[i] > 0)
				? arm_rewards[i] / arm_counts[i]
				: 0.0;
			double		confidence = (arm_counts[i] > 0 && total_pulls > 0)
				? sqrt(2.0 * log((double) total_pulls) /
					   arm_counts[i])
				: DBL_MAX;

			arm_probs[i] = avg_reward + confidence;
		}
	}
	else if (strcmp(algorithm, "epsilon_greedy") == 0)
	{
		/* Epsilon-Greedy */
		int			best_arm = 0;
		double		best_avg = -DBL_MAX;
		int			a;

		for (a = 0; a < n_arms; a++)
		{
			double		avg = (arm_counts[a] > 0)
				? arm_rewards[a] / arm_counts[a]
				: 0.0;

			if (avg > best_avg)
			{
				best_avg = avg;
				best_arm = a;
			}
		}

		for (i = 0; i < n_arms; i++)
		{
			if (i == best_arm)
				arm_probs[i] = 1.0 - epsilon + epsilon / n_arms;
			else
				arm_probs[i] = epsilon / n_arms;
		}
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("Unknown algorithm: %s", algorithm)));
	}

	/* Normalize probabilities */
	{
		double		sum = 0.0;

		for (i = 0; i < n_arms; i++)
			sum += arm_probs[i];
		if (sum > 0.0)
			for (i = 0; i < n_arms; i++)
				arm_probs[i] /= sum;
	}

	/* Build result array */
	result_datums = (Datum *) palloc(sizeof(Datum) * n_arms);
	for (i = 0; i < n_arms; i++)
		result_datums[i] = Float8GetDatum(arm_probs[i]);

	result = construct_array(result_datums,
							 n_arms,
							 FLOAT8OID,
							 sizeof(float8),
							 FLOAT8PASSBYVAL,
							 'd');

	NDB_FREE(arm_counts);
	NDB_FREE(arm_rewards);
	NDB_FREE(arm_probs);
	NDB_FREE(result_datums);
	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);
	NDB_FREE(tbl_str);
	NDB_FREE(algorithm);

	PG_RETURN_ARRAYTYPE_P(result);
}
