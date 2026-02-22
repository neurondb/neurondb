/*-------------------------------------------------------------------------
 *
 * ml_neural_network.c
 *    Feedforward neural network implementation.
 *
 * This module implements feedforward neural networks with backpropagation,
 * supporting multiple hidden layers and various activation functions.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_neural_network.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/builtins.h"
#include "utils/array.h"
#include "utils/jsonb.h"
#include "neurondb_json.h"
#include "executor/spi.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "utils/memutils.h"
#include "neurondb_pgcompat.h"
#include "neurondb.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_spi.h"
#include "neurondb_macros.h"
#include "ml_catalog.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Neural Network Structures */
typedef struct NeuralLayer
{
	int			n_inputs;
	int			n_outputs;
	float	  **weights;		/* [n_outputs][n_inputs+1] (includes bias) */
	float	   *activations;	/* Current layer activations */
	float	   *deltas;			/* Backpropagation deltas */
}			NeuralLayer;

typedef struct NeuralNetwork
{
	int			n_layers;
	int			n_inputs;
	int			n_outputs;
	NeuralLayer *layers;
	char	   *activation_func;	/* "relu", "sigmoid", "tanh" */
	float		learning_rate;
}			NeuralNetwork;

/* Activation functions */
static float
activation_relu(float x)
{
	return (x > 0.0f) ? x : 0.0f;
}

static float
activation_sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static float
activation_tanh(float x)
{
	return tanhf(x);
}

static float
activation_derivative_relu(float x)
{
	return (x > 0.0f) ? 1.0f : 0.0f;
}

static float
activation_derivative_sigmoid(float x)
{
	float		s = activation_sigmoid(x);

	return s * (1.0f - s);
}

static float
activation_derivative_tanh(float x)
{
	float		t = tanhf(x);

	return 1.0f - t * t;
}

/*
 * neural_network_forward - Execute forward propagation through all network layers
 *
 * Processes input data through each layer of the network, computing weighted
 * sums, applying activation functions, and propagating activations to subsequent
 * layers. Validates all inputs and intermediate results for numerical stability.
 *
 * Parameters:
 *   net - Neural network structure with layers and weights
 *   input - Input feature vector
 *   output - Output vector to store final predictions
 *
 * Notes:
 *   Each layer receives activations from the previous layer, applies matrix
 *   multiplication with learned weights, adds bias terms, and transforms through
 *   an activation function (ReLU, sigmoid, tanh, or linear). The function
 *   performs extensive validation to check for NULL pointers, invalid layer
 *   counts, and non-finite values that could indicate numerical overflow.
 */
static void
neural_network_forward(NeuralNetwork * net, float *input, float *output)
{
	int			i,
				j,
				k;
	float	   *prev_activations = input;
	float *curr_activations = NULL;

	if (net == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neural_network_forward: network cannot be NULL")));

	if (input == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neural_network_forward: input cannot be NULL")));

	if (output == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neural_network_forward: output cannot be NULL")));

	if (net->layers == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neural_network_forward: network layers are NULL")));

	if (net->n_layers <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neural_network_forward: invalid n_layers %d",
						net->n_layers)));

	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];

		if (layer == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neural_network_forward: layer %d is NULL", i)));

		if (layer->weights == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neural_network_forward: layer %d weights are NULL",
							i)));

		if (layer->activations == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neural_network_forward: layer %d activations are NULL",
							i)));

		curr_activations = layer->activations;

		for (j = 0; j < layer->n_outputs; j++)
		{
			float		sum;

			if (layer->weights[j] == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neural_network_forward: layer %d, output %d weights are NULL",
								i, j)));

			sum = layer->weights[j][layer->n_inputs];	/* bias */

			for (k = 0; k < layer->n_inputs; k++)
			{
				if (!isfinite(prev_activations[k]))
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neural_network_forward: non-finite input at layer %d, dimension %d",
									i, k)));
				if (!isfinite(layer->weights[j][k]))
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neural_network_forward: non-finite weight at layer %d, output %d, input %d",
									i, j, k)));
				sum += layer->weights[j][k] * prev_activations[k];
			}

			/* Check for overflow/underflow */
			if (!isfinite(sum))
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("neural_network_forward: non-finite sum at layer %d, output %d",
								i, j)));

			/* Apply activation */
			if (net->activation_func != NULL)
			{
				if (strcmp(net->activation_func, "relu") == 0)
					curr_activations[j] = activation_relu(sum);
				else if (strcmp(net->activation_func, "sigmoid") == 0)
					curr_activations[j] = activation_sigmoid(sum);
				else if (strcmp(net->activation_func, "tanh") == 0)
					curr_activations[j] = activation_tanh(sum);
				else
					curr_activations[j] = sum;	/* linear */
			}
			else
			{
				curr_activations[j] = sum;	/* linear (fallback) */
			}

			if (!isfinite(curr_activations[j]))
				ereport(ERROR,
						(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
						 errmsg("neural_network_forward: non-finite activation at layer %d, output %d",
								i, j)));
		}

		prev_activations = curr_activations;
	}

	/* Copy final layer to output */
	if (net->n_layers > 0 && net->layers[net->n_layers - 1].activations != NULL)
	{
		memcpy(output,
			   net->layers[net->n_layers - 1].activations,
			   net->n_outputs * sizeof(float));
	}
	else
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neural_network_forward: invalid final layer")));
	}
}

/*
 * neural_network_backward
 *    Execute backpropagation algorithm to compute gradients for weight updates.
 *
 * This function implements the backpropagation algorithm, computing gradients
 * of the loss function with respect to all network weights. The algorithm
 * begins at the output layer by calculating the error between predicted and
 * target values, then multiplies this error by the derivative of the
 * activation function to obtain output layer deltas. These deltas are then
 * propagated backward through hidden layers using the chain rule of
 * calculus. For each hidden layer, the gradient flows from the next layer
 * through the weights connecting the layers, multiplied by the derivative
 * of the current layer's activation function. The computed deltas represent
 * how much each neuron's output contributes to the overall error, and they
 * are used to update weights during training to minimize the loss function.
 * This process enables the network to learn from training examples by
 * adjusting weights in the direction that reduces prediction error.
 */
static void
neural_network_backward(NeuralNetwork * net,
						float *input,
						float *target,
						float *predicted)
{
	int			i,
				j,
				k;
	NeuralLayer *output_layer = &net->layers[net->n_layers - 1];

	/* Compute output layer deltas */
	for (j = 0; j < output_layer->n_outputs; j++)
	{
		float		error = target[j] - predicted[j];
		float		activation = output_layer->activations[j];
		float		derivative;

		if (strcmp(net->activation_func, "relu") == 0)
			derivative = activation_derivative_relu(activation);
		else if (strcmp(net->activation_func, "sigmoid") == 0)
			derivative = activation_derivative_sigmoid(activation);
		else if (strcmp(net->activation_func, "tanh") == 0)
			derivative = activation_derivative_tanh(activation);
		else
			derivative = 1.0f;

		output_layer->deltas[j] = error * derivative;
	}

	/* Backpropagate through hidden layers */
	for (i = net->n_layers - 2; i >= 0; i--)
	{
		NeuralLayer *curr_layer = &net->layers[i];
		NeuralLayer *next_layer = &net->layers[i + 1];

		for (j = 0; j < curr_layer->n_outputs; j++)
		{
			float		sum = 0.0f;
			float		activation;
			float		derivative;

			for (k = 0; k < next_layer->n_outputs; k++)
				sum += next_layer->weights[k][j]
					* next_layer->deltas[k];

			activation = curr_layer->activations[j];

			if (strcmp(net->activation_func, "relu") == 0)
				derivative =
					activation_derivative_relu(activation);
			else if (strcmp(net->activation_func, "sigmoid") == 0)
				derivative = activation_derivative_sigmoid(
														   activation);
			else if (strcmp(net->activation_func, "tanh") == 0)
				derivative =
					activation_derivative_tanh(activation);
			else
				derivative = 1.0f;

			curr_layer->deltas[j] = sum * derivative;
		}
	}
}

/* Update weights using gradient descent */
static void
neural_network_update_weights(NeuralNetwork * net, float *input)
{
	int			i,
				j,
				k;
	float	   *prev_activations = input;

	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];

		for (j = 0; j < layer->n_outputs; j++)
		{
			/* Update bias */
			layer->weights[j][layer->n_inputs] +=
				net->learning_rate * layer->deltas[j];

			/* Update input weights */
			for (k = 0; k < layer->n_inputs; k++)
				layer->weights[j][k] += net->learning_rate
					* layer->deltas[j]
					* prev_activations[k];
		}

		prev_activations = layer->activations;
	}
}

/* Initialize neural network */
static NeuralNetwork *
neural_network_init(int n_inputs,
					int n_outputs,
					int *hidden_layers,
					int n_hidden,
					const char *activation,
					float learning_rate)
{
	int			i;
	int			j;
	int			k;
	int			prev_size;
	NeuralLayer *output_layer = NULL;
	NeuralNetwork *net = NULL;
	NeuralLayer *layers = NULL;

	nalloc(net, NeuralNetwork, 1);

	net->n_inputs = n_inputs;
	net->n_outputs = n_outputs;
	net->n_layers = n_hidden + 1;	/* hidden + output */
	net->activation_func = pstrdup(activation);
	net->learning_rate = learning_rate;
	nalloc(layers, NeuralLayer, net->n_layers);
	net->layers = layers;

	/* Initialize hidden layers */
	prev_size = n_inputs;

	for (i = 0; i < n_hidden; i++)
	{
		NeuralLayer *layer = &net->layers[i];
		float	  **weights;
		float *activations = NULL;
		float *deltas = NULL;

		layer->n_inputs = prev_size;
		layer->n_outputs = hidden_layers[i];
		nalloc(weights, float *, layer->n_outputs);
		nalloc(activations, float, layer->n_outputs);
		nalloc(deltas, float, layer->n_outputs);
		layer->weights = weights;
		layer->activations = activations;
		layer->deltas = deltas;

		for (j = 0; j < layer->n_outputs; j++)
		{
			float *weight_row = NULL;
			nalloc(weight_row, float, layer->n_inputs + 1);
			layer->weights[j] = weight_row;
			/* Initialize weights randomly (small values) */
			for (k = 0; k <= layer->n_inputs; k++)
				layer->weights[j][k] =
					(float) (((double) rand() / (double) RAND_MAX)) * 0.1f
					- 0.05f;
		}

		prev_size = layer->n_outputs;
	}

	/* Initialize output layer */
	output_layer = &net->layers[n_hidden];
	output_layer->n_inputs = prev_size;
	output_layer->n_outputs = n_outputs;
	{
		float	  **output_weights = NULL;
		float *output_activations = NULL;
		float *output_deltas = NULL;

		nalloc(output_weights, float *, output_layer->n_outputs);
		nalloc(output_activations, float, output_layer->n_outputs);
		nalloc(output_deltas, float, output_layer->n_outputs);
		output_layer->weights = output_weights;
		output_layer->activations = output_activations;
		output_layer->deltas = output_deltas;

		for (j = 0; j < output_layer->n_outputs; j++)
		{
			float *output_weight_row = NULL;

			nalloc(output_weight_row, float, output_layer->n_inputs + 1);
			output_layer->weights[j] = output_weight_row;
			for (k = 0; k <= output_layer->n_inputs; k++)
				output_layer->weights[j][k] =
					(float) (((double) rand() / (double) RAND_MAX)) * 0.1f - 0.05f;
		}
	}

	return net;
}

/* Free neural network */
static void
neural_network_free(NeuralNetwork * net)
{
	int			i,
				j;

	if (!net)
		return;

	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];

		for (j = 0; j < layer->n_outputs; j++)
			nfree(layer->weights[j]);

		nfree(layer->weights);
		nfree(layer->activations);
		nfree(layer->deltas);
	}

	nfree(net->layers);
	nfree(net->activation_func);
	nfree(net);
}

/* Serialize neural network to bytea */
static bytea *
neural_network_serialize(const NeuralNetwork * net, uint8 training_backend)
{
	StringInfoData buf;
	int			i,
				j,
				k;
	int			activation_len;

	if (net == NULL)
		return NULL;

	/* Validate training_backend */
	if (training_backend > 1)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: neural_network_serialize: invalid training_backend %d (must be 0 or 1)",
						training_backend)));
	}

	/* Validate model before serialization */
	if (net->n_layers <= 0 || net->n_layers > 100)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: neural_network_serialize: invalid n_layers %d",
						net->n_layers)));
	}

	if (net->n_inputs <= 0 || net->n_inputs > 10000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: neural_network_serialize: invalid n_inputs %d",
						net->n_inputs)));
	}

	if (net->n_outputs <= 0 || net->n_outputs > 1000)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: neural_network_serialize: invalid n_outputs %d",
						net->n_outputs)));
	}

	if (net->activation_func == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: neural_network_serialize: activation_func is NULL")));
	}

	pq_begintypsend(&buf);

	/* Write training_backend first (0=CPU, 1=GPU) - unified storage format */
	pq_sendbyte(&buf, training_backend);
	/* Write header */
	pq_sendint32(&buf, net->n_layers);
	pq_sendint32(&buf, net->n_inputs);
	pq_sendint32(&buf, net->n_outputs);
	pq_sendfloat8(&buf, net->learning_rate);

	/* Write activation function */
	activation_len = strlen(net->activation_func);
	if (activation_len > 255)
		activation_len = 255;
	pq_sendint32(&buf, activation_len);
	pq_sendbytes(&buf, net->activation_func, activation_len);

	/* Write each layer */
	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];

		if (layer == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: neural_network_serialize: layer %d is NULL",
							i)));
		}

		/* Validate layer dimensions before serialization */
		if (layer->n_inputs <= 0 || layer->n_inputs > 10000
			|| layer->n_outputs <= 0 || layer->n_outputs > 10000)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: neural_network_serialize: invalid layer %d dimensions (%d, %d)",
							i, layer->n_inputs, layer->n_outputs)));
		}

		pq_sendint32(&buf, layer->n_inputs);
		pq_sendint32(&buf, layer->n_outputs);

		/* Write weights matrix [n_outputs][n_inputs+1] (includes bias) */
		if (layer->weights == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: neural_network_serialize: layer %d weights are NULL",
							i)));
		}

		if (layer->n_outputs > 0 && layer->n_inputs > 0)
		{
			for (j = 0; j < layer->n_outputs; j++)
			{
				if (layer->weights[j] == NULL)
				{
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: neural_network_serialize: layer %d, output %d weights are NULL",
									i, j)));
				}

				for (k = 0; k <= layer->n_inputs; k++)
				{
					if (!isfinite(layer->weights[j][k]))
					{
						ereport(ERROR,
								(errcode(ERRCODE_INTERNAL_ERROR),
								 errmsg("neurondb: neural_network_serialize: non-finite weight at layer %d, output %d, input %d",
										i, j, k)));
					}
					pq_sendfloat8(&buf, layer->weights[j][k]);
				}
			}
		}
	}

	return pq_endtypsend(&buf);
}

/* Deserialize neural network from bytea */
static NeuralNetwork *
neural_network_deserialize(const bytea * data, uint8 * training_backend_out)
{
	NeuralNetwork *net = NULL;
	StringInfoData buf;
	int			i,
				j,
				k;
	int			activation_len;
	char *activation_buf = NULL;
	uint8		training_backend = 0;

	if (data == NULL)
		return NULL;

	/* Validate bytea size */
	if (VARSIZE(data) < VARHDRSZ + sizeof(uint8))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neural_network_deserialize: invalid bytea size")));
	}

	buf.data = VARDATA(data);
	buf.len = VARSIZE(data) - VARHDRSZ;
	buf.maxlen = buf.len;
	buf.cursor = 0;

	/* Check minimum size for header */
	if (buf.len < sizeof(uint8) + 4 * 4 + 8 + 4)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neural_network_deserialize: bytea too small for header")));
	}

	nalloc(net, NeuralNetwork, 1);

	/* Read training_backend first (unified storage format) */
	training_backend = (uint8) pq_getmsgint(&buf, 1);
	if (training_backend_out != NULL)
		*training_backend_out = training_backend;
	/* Read header */
	net->n_layers = pq_getmsgint(&buf, 4);
	net->n_inputs = pq_getmsgint(&buf, 4);
	net->n_outputs = pq_getmsgint(&buf, 4);
	net->learning_rate = pq_getmsgfloat8(&buf);

	/* Validate deserialized values */
	if (net->n_layers <= 0 || net->n_layers > 100)
	{
		nfree(net);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neural_network_deserialize: invalid n_layers %d",
						net->n_layers)));
	}

	if (net->n_inputs <= 0 || net->n_inputs > 10000)
	{
		nfree(net);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neural_network_deserialize: invalid n_inputs %d",
						net->n_inputs)));
	}

	if (net->n_outputs <= 0 || net->n_outputs > 1000)
	{
		nfree(net);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neural_network_deserialize: invalid n_outputs %d",
						net->n_outputs)));
	}

	/* Read activation function */
	activation_len = pq_getmsgint(&buf, 4);
	if (activation_len < 0 || activation_len > 255)
	{
		nfree(net);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neural_network_deserialize: invalid activation_len %d",
						activation_len)));
	}

	/* Check buffer has enough data for activation string */
	if (buf.cursor + activation_len > buf.len)
	{
		nfree(net);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: neural_network_deserialize: buffer overflow reading activation function")));
	}

	nalloc(activation_buf, char, activation_len + 1);
	pq_copymsgbytes(&buf, activation_buf, activation_len);
	activation_buf[activation_len] = '\0';
	net->activation_func = activation_buf;

	/* Allocate layers with overflow check */
	{
		NeuralLayer *layers = NULL;
		if (net->n_layers > MaxAllocSize / sizeof(NeuralLayer))
		{
			nfree(net->activation_func);
			nfree(net);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb: neural_network_deserialize: n_layers %d exceeds maximum allocation size",
							net->n_layers)));
		}
		nalloc(layers, NeuralLayer, net->n_layers);
		net->layers = layers;
	}

	/* Read each layer */
	for (i = 0; i < net->n_layers; i++)
	{
		NeuralLayer *layer = &net->layers[i];
		size_t		weights_size;
		size_t		activations_size;
		size_t		deltas_size;
		size_t		weight_row_size;
		float	  **layer_weights;
		float *layer_activations = NULL;
		float *layer_deltas = NULL;

		layer->n_inputs = pq_getmsgint(&buf, 4);
		layer->n_outputs = pq_getmsgint(&buf, 4);

		/* Validate layer dimensions */
		if (layer->n_inputs <= 0 || layer->n_inputs > 10000
			|| layer->n_outputs <= 0 || layer->n_outputs > 10000)
		{
			/* Cleanup allocated layers so far */
			for (j = 0; j < i; j++)
			{
				NeuralLayer *prev_layer = &net->layers[j];

				if (prev_layer->weights != NULL)
				{
					for (k = 0; k < prev_layer->n_outputs; k++)
						if (prev_layer->weights[k] != NULL)
							nfree(prev_layer->weights[k]);
					nfree(prev_layer->weights);
				}
				if (prev_layer->activations != NULL)
					nfree(prev_layer->activations);
				if (prev_layer->deltas != NULL)
					nfree(prev_layer->deltas);
			}
			nfree(net->layers);
			nfree(net->activation_func);
			nfree(net);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: neural_network_deserialize: invalid layer %d dimensions (%d, %d)",
							i, layer->n_inputs, layer->n_outputs)));
		}

		/* Check for integer overflow in allocations */
		weights_size = (size_t) layer->n_outputs * sizeof(float *);
		activations_size = (size_t) layer->n_outputs * sizeof(float);
		deltas_size = (size_t) layer->n_outputs * sizeof(float);
		weight_row_size = (size_t) (layer->n_inputs + 1) * sizeof(float);

		if (weights_size > MaxAllocSize
			|| activations_size > MaxAllocSize
			|| deltas_size > MaxAllocSize
			|| weight_row_size > MaxAllocSize)
		{
			/* Cleanup allocated layers so far */
			for (j = 0; j < i; j++)
			{
				NeuralLayer *prev_layer = &net->layers[j];

				if (prev_layer->weights != NULL)
				{
					for (k = 0; k < prev_layer->n_outputs; k++)
						if (prev_layer->weights[k] != NULL)
							nfree(prev_layer->weights[k]);
					nfree(prev_layer->weights);
				}
				if (prev_layer->activations != NULL)
					nfree(prev_layer->activations);
				if (prev_layer->deltas != NULL)
					nfree(prev_layer->deltas);
			}
			nfree(net->layers);
			nfree(net->activation_func);
			nfree(net);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("neurondb: neural_network_deserialize: layer %d allocation size exceeds maximum",
							i)));
		}

		/* Allocate layer arrays */
		nalloc(layer_weights, float *, layer->n_outputs);
		nalloc(layer_activations, float, layer->n_outputs);
		nalloc(layer_deltas, float, layer->n_outputs);
		layer->weights = layer_weights;
		layer->activations = layer_activations;
		layer->deltas = layer_deltas;

		/* Check buffer has enough data for weights */
		{
			size_t		weights_bytes_needed =
				(size_t) layer->n_outputs * (size_t) (layer->n_inputs + 1) * 8;

			if (buf.cursor + weights_bytes_needed > buf.len)
			{
				for (j = 0; j < i; j++)
				{
					NeuralLayer *prev_layer = &net->layers[j];

					if (prev_layer->weights != NULL)
					{
						for (k = 0; k < prev_layer->n_outputs; k++)
							if (prev_layer->weights[k] != NULL)
								nfree(prev_layer->weights[k]);
						nfree(prev_layer->weights);
					}
					if (prev_layer->activations != NULL)
						nfree(prev_layer->activations);
					if (prev_layer->deltas != NULL)
						nfree(prev_layer->deltas);
				}
				nfree(net->layers);
				nfree(net->activation_func);
				nfree(net);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: neural_network_deserialize: buffer overflow reading layer %d weights",
								i)));
			}
		}

		/* Read weights matrix */
		for (j = 0; j < layer->n_outputs; j++)
		{
			float *weight_row = NULL;
			
			PG_TRY();
			{
				nalloc(weight_row, float, layer->n_inputs + 1);
				layer->weights[j] = weight_row;
			}
			PG_CATCH();
			{
				/* Cleanup on allocation failure */
				for (k = 0; k < j; k++)
					if (layer->weights[k] != NULL)
						nfree(layer->weights[k]);
				if (layer->weights != NULL)
					nfree(layer->weights);
				if (layer->activations != NULL)
					nfree(layer->activations);
				if (layer->deltas != NULL)
					nfree(layer->deltas);
				for (k = 0; k < i; k++)
				{
					NeuralLayer *prev_layer = &net->layers[k];

					if (prev_layer->weights != NULL)
					{
						for (int m = 0; m < prev_layer->n_outputs; m++)
							if (prev_layer->weights[m] != NULL)
								nfree(prev_layer->weights[m]);
						nfree(prev_layer->weights);
					}
					if (prev_layer->activations != NULL)
						nfree(prev_layer->activations);
					if (prev_layer->deltas != NULL)
						nfree(prev_layer->deltas);
				}
				nfree(net->layers);
				nfree(net->activation_func);
				nfree(net);
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("neurondb: neural_network_deserialize: failed to allocate weights for layer %d, output %d",
								i, j)));
			}
			PG_END_TRY();
			for (k = 0; k <= layer->n_inputs; k++)
			{
				float		weight_val = (float) pq_getmsgfloat8(&buf);

				if (!isfinite(weight_val))
				{
					for (k = 0; k <= j; k++)
						nfree(layer->weights[k]);
					nfree(layer->weights);
					nfree(layer->activations);
					nfree(layer->deltas);
					for (j = 0; j < i; j++)
					{
						NeuralLayer *prev_layer = &net->layers[j];

						if (prev_layer->weights != NULL)
						{
							for (k = 0; k < prev_layer->n_outputs; k++)
								if (prev_layer->weights[k] != NULL)
									nfree(prev_layer->weights[k]);
							nfree(prev_layer->weights);
						}
						if (prev_layer->activations != NULL)
							nfree(prev_layer->activations);
						if (prev_layer->deltas != NULL)
							nfree(prev_layer->deltas);
					}
					nfree(net->layers);
					nfree(net->activation_func);
					nfree(net);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neurondb: neural_network_deserialize: non-finite weight at layer %d, output %d, input %d",
									i, j, k)));
				}
				layer->weights[j][k] = weight_val;
			}
		}
	}

	return net;
}

/*
 * Train neural network
 *
 * train_neural_network(
 *   table_name text,
 *   feature_col text,
 *   label_col text,
 *   hidden_layers int[],
 *   activation text DEFAULT 'relu',
 *   learning_rate float8 DEFAULT 0.01,
 *   epochs int DEFAULT 100,
 *   batch_size int DEFAULT 32
 * )
 */
/*
 * train_neural_network - Train a feedforward neural network
 *
 * User-facing function that trains a feedforward neural network on data from
 * a table using backpropagation. Supports multiple hidden layers and various
 * activation functions.
 *
 * Parameters:
 *   table_name - Name of table containing training data (text)
 *   feature_col - Name of feature column (text)
 *   label_col - Name of label column (text)
 *   hidden_layers - Array of hidden layer sizes (int32[])
 *   activation - Activation function name: "relu", "sigmoid", or "tanh" (text)
 *   learning_rate - Learning rate for gradient descent (float8)
 *   max_iters - Maximum training iterations (int32, optional)
 *
 * Returns:
 *   Model ID (int32) of the trained model stored in catalog
 *
 * Notes:
 *   The function uses backpropagation with the specified learning rate to
 *   train the network. Supports ReLU, sigmoid, and tanh activation functions.
 *   The trained model is serialized and stored in the ML catalog.
 */
PG_FUNCTION_INFO_V1(train_neural_network);

Datum
train_neural_network(PG_FUNCTION_ARGS)
{
	text	   *table_name = PG_GETARG_TEXT_PP(0);
	text	   *feature_col = PG_GETARG_TEXT_PP(1);
	text	   *label_col = PG_GETARG_TEXT_PP(2);
	ArrayType  *hidden_layers_array = PG_GETARG_ARRAYTYPE_P(3);
	text	   *activation_text = PG_ARGISNULL(4) ? NULL : PG_GETARG_TEXT_PP(4);
	float8		learning_rate = PG_ARGISNULL(5) ? 0.01 : PG_GETARG_FLOAT8(5);
	int32		epochs = PG_ARGISNULL(6) ? 100 : PG_GETARG_INT32(6);
	char *table_name_str = NULL;
	char *feature_col_str = NULL;
	char *label_col_str = NULL;
	char *activation = NULL;
	int			n_hidden;
	int *hidden_layers = NULL;
	int			n_inputs = 0;
	int			n_outputs = 1;
	MemoryContext oldcontext;
	MemoryContext callcontext;
	StringInfoData sql;
	int			ret;
	StringInfoData hyperbuf;
	SPITupleTable *tuptable = NULL;
	TupleDesc	tupdesc;
	int			n_samples = 0;
	bytea *serialized = NULL;
	Jsonb *params_jsonb = NULL;
	Jsonb *metrics_jsonb = NULL;
	float **X = NULL;

	float *y = NULL;
	NeuralNetwork *net = NULL;
	int			epoch,
				sample;
	float		loss;
	int			i;
	int			j;
	StringInfoData metricsbuf;
	MLCatalogModelSpec spec;
	int32		model_id = 0;

	char *hidden_layers_json = NULL;
	int			idx;

	table_name_str = text_to_cstring(table_name);
	feature_col_str = text_to_cstring(feature_col);
	label_col_str = text_to_cstring(label_col);
	activation = activation_text ? text_to_cstring(activation_text)
		: pstrdup("relu");

	/* Get batch_size parameter */
	{
		int32		batch_size = PG_ARGISNULL(7) ? 32 : PG_GETARG_INT32(7);

		if (batch_size < 1 || batch_size > n_samples)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("batch_size must be between 1 and number of samples")));

		/* Validate activation function */
		if (strcmp(activation, "relu") != 0
			&& strcmp(activation, "sigmoid") != 0
			&& strcmp(activation, "tanh") != 0
			&& strcmp(activation, "linear") != 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("activation must be 'relu', 'sigmoid', "
							"'tanh', or 'linear'")));

		/* Extract hidden layers */
		n_hidden = ArrayGetNItems(
								  ARR_NDIM(hidden_layers_array), ARR_DIMS(hidden_layers_array));
		nalloc(hidden_layers, int, n_hidden);

		for (i = 0; i < n_hidden; i++)
		{
			bool		isnull;
			Datum		elem;

			elem = array_ref(hidden_layers_array,
							 1,
							 &i,
							 -1,
							 -1,
							 false,
							 'i',
							 &isnull);

			if (isnull)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("hidden_layers array cannot "
								"contain NULL")));

			hidden_layers[i] = DatumGetInt32(elem);
			if (hidden_layers[i] <= 0)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("hidden layer sizes must be "
								"positive")));
		}

		/* Create memory context */
		callcontext = AllocSetContextCreate(CurrentMemoryContext,
											"train_neural_network memory context",
											ALLOCSET_DEFAULT_SIZES);
		oldcontext = MemoryContextSwitchTo(callcontext);

		/* Initialize and build query in callcontext BEFORE SPI_connect */
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
						 quote_identifier(feature_col_str),
						 quote_identifier(label_col_str),
						 quote_identifier(table_name_str),
						 quote_identifier(feature_col_str),
						 quote_identifier(label_col_str));

		{
			NdbSpiSession *train_nn_spi_session = NULL;

			NDB_SPI_SESSION_BEGIN(train_nn_spi_session, oldcontext);

			ret = ndb_spi_execute_safe(sql.data, true, 0);
			NDB_CHECK_SPI_TUPTABLE();
			if (ret != SPI_OK_SELECT)
			{
				nfree(sql.data);
				NDB_SPI_SESSION_END(train_nn_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: train_neural_network: failed to load training data"),
						 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
						 errhint("Verify the table exists and contains valid feature and label columns.")));
			}

			tuptable = SPI_tuptable;
			tupdesc = tuptable->tupdesc;
			n_samples = SPI_processed;

			if (n_samples == 0)
			{
				nfree(sql.data);
				NDB_SPI_SESSION_END(train_nn_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neurondb: train_neural_network: no training data found"),
						 errdetail("The query returned 0 rows"),
						 errhint("Ensure the table contains valid data with non-NULL features and labels.")));
			}

			/* Determine input/output dimensions will be set from first vector */
			n_inputs = 0;		/* Will be determined from first vector */
			n_outputs = 1;		/* Regression - single output */

			/* Determine actual feature dimension from first row */
			if (n_samples > 0)
			{
				HeapTuple	first_tuple = tuptable->vals[0];
				bool		isnull;
				Datum		feat_datum;

				feat_datum = SPI_getbinval(first_tuple, tupdesc, 1, &isnull);
				if (isnull)
					ereport(ERROR,
							(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
							 errmsg("neural network training: feature vector cannot be NULL")));

				/* Extract dimension from vector type */
				/* Check if type is vector by trying to cast to Vector */
				{
					Vector *test_vec = NULL;

					test_vec = DatumGetVector(feat_datum);
					if (test_vec != NULL && test_vec->dim > 0)
					{
						n_inputs = test_vec->dim;
						if (n_inputs <= 0 || n_inputs > 10000)
							ereport(ERROR,
									(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
									 errmsg("neural network training: invalid vector dimension %d",
											n_inputs)));
					}
					else
					{
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neural network training: feature column must be vector type")));
					}
				}
			}
			else
			{
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("neural network training: no training data found")));
			}

			/* Allocate training data arrays with correct dimensions */
			nalloc(X, float *, n_samples);
			nalloc(y, float, n_samples);

			for (i = 0; i < n_samples; i++)
			{
				HeapTuple	tuple = tuptable->vals[i];
				bool		isnull;
				Datum		feat_datum;
				Datum		label_datum;
				Vector *vec = NULL;

				/* Extract feature vector */
				feat_datum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
				if (isnull)
				{
					for (j = 0; j < i; j++)
						nfree(X[j]);
					nfree(X);
					nfree(y);
					ereport(ERROR,
							(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
							 errmsg("neural network training: feature vector at row %d cannot be NULL",
									i + 1)));
				}

				vec = DatumGetVector(feat_datum);
				if (vec == NULL || vec->dim != n_inputs)
				{
					for (j = 0; j < i; j++)
						nfree(X[j]);
					nfree(X);
					nfree(y);
					ereport(ERROR,
							(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
							 errmsg("neural network training: invalid vector at row %d (expected dim %d, got %d)",
									i + 1, n_inputs, vec ? vec->dim : 0)));
				}

				/* Copy vector data to feature matrix */
				nalloc(X[i], float, n_inputs);
				for (j = 0; j < n_inputs; j++)
				{
					if (!isfinite(vec->data[j]))
					{
						for (j = 0; j <= i; j++)
							nfree(X[j]);
						nfree(X);
						nfree(y);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neural network training: non-finite value in feature vector at row %d, dimension %d",
										i + 1, j)));
					}
					X[i][j] = vec->data[j];
				}

				/* Extract label */
				label_datum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
				if (isnull)
				{
					for (j = 0; j <= i; j++)
						nfree(X[j]);
					nfree(X);
					nfree(y);
					ereport(ERROR,
							(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
							 errmsg("neural network training: label at row %d cannot be NULL",
									i + 1)));
				}

				{
					Oid			label_type = SPI_gettypeid(tupdesc, 2);

					if (label_type == FLOAT8OID)
						y[i] = (float) DatumGetFloat8(label_datum);
					else if (label_type == FLOAT4OID)
						y[i] = DatumGetFloat4(label_datum);
					else if (label_type == INT4OID)
						y[i] = (float) DatumGetInt32(label_datum);
					else if (label_type == INT8OID)
						y[i] = (float) DatumGetInt64(label_datum);
					else
					{
						for (j = 0; j <= i; j++)
							nfree(X[j]);
						nfree(X);
						nfree(y);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neural network training: unsupported label type")));
					}

					if (!isfinite(y[i]))
					{
						for (j = 0; j <= i; j++)
							nfree(X[j]);
						nfree(X);
						nfree(y);
						ereport(ERROR,
								(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
								 errmsg("neural network training: non-finite label value at row %d",
										i + 1)));
					}
				}
			}

			/* Initialize neural network */
			net = neural_network_init(n_inputs,
									  n_outputs,
									  hidden_layers,
									  n_hidden,
									  activation,
									  (float) learning_rate);

			/* Training loop with batch processing */
			for (epoch = 0; epoch < epochs; epoch++)
			{
				loss = 0.0f;
				{
					int			batch_start = 0;

					/* Process samples in batches */
					while (batch_start < n_samples)
					{
						int			batch_end;
						int			batch_count;

						batch_end = batch_start + batch_size;
						if (batch_end > n_samples)
							batch_end = n_samples;
						batch_count = batch_end - batch_start;

						/* Process each sample in batch */
						for (sample = batch_start; sample < batch_end; sample++)
						{
							float		predicted[1];
							float		error;
							float		target[1];

							/* Forward pass */
							neural_network_forward(net, X[sample], predicted);

							/* Compute loss */
							error = y[sample] - predicted[0];
							loss += error * error;

							/* Backward pass (computes gradients) */
							target[0] = y[sample];
							neural_network_backward(
													net, X[sample], target, predicted);

							/* For batch training, we accumulate gradients */

							/*
							 * Since neural_network_backward overwrites deltas
							 * each time,
							 */
							/* we update weights after processing the batch */
						}

						/* Update weights once per batch */
						/* Use average gradient by scaling learning rate */
						if (batch_count > 0)
						{
							float		original_lr = net->learning_rate;

							net->learning_rate = original_lr / (float) batch_count;

							/*
							 * Update weights using gradients from last sample
							 * in batch
							 */

							/*
							 * (gradients are already computed, we just scale
							 * the update)
							 */
							neural_network_update_weights(net, X[batch_end - 1]);

							/* Restore original learning rate */
							net->learning_rate = original_lr;
						}

						batch_start = batch_end;
					}
				}

				loss /= n_samples;

				if (epoch % 10 == 0)
				{
				}
			}

			/* Serialize neural network with error handling */
			PG_TRY();
			{
				serialized = neural_network_serialize(net, 0); /* training_backend=0 for CPU */
				if (serialized == NULL)
				{
					nfree(sql.data);
					NDB_SPI_SESSION_END(train_nn_spi_session);
					MemoryContextSwitchTo(oldcontext);
					MemoryContextDelete(callcontext);
					for (i = 0; i < n_samples; i++)
						nfree(X[i]);
					nfree(X);
					nfree(y);
					neural_network_free(net);
					nfree(hidden_layers);
					nfree(table_name_str);
					nfree(feature_col_str);
					nfree(label_col_str);
					nfree(activation);
					ereport(ERROR,
							(errcode(ERRCODE_INTERNAL_ERROR),
							 errmsg("neurondb: train_neural_network: failed to serialize model")));
				}
			}
			PG_CATCH();
			{
				/* Cleanup on serialization error */
				nfree(sql.data);
				NDB_SPI_SESSION_END(train_nn_spi_session);
				MemoryContextSwitchTo(oldcontext);
				MemoryContextDelete(callcontext);
				for (i = 0; i < n_samples; i++)
					nfree(X[i]);
				nfree(X);
				nfree(y);
				neural_network_free(net);
				nfree(hidden_layers);
				nfree(table_name_str);
				nfree(feature_col_str);
				nfree(label_col_str);
				nfree(activation);
				PG_RE_THROW();
			}
			PG_END_TRY();

			/* Build hyperparameters JSON */
			initStringInfo(&hyperbuf);
			/* Estimate JSON length: [num1,num2,...] */
			if (n_hidden > 0)
			{
				StringInfoData layerbuf;

				initStringInfo(&layerbuf);
				appendStringInfoChar(&layerbuf, '[');
				for (idx = 0; idx < n_hidden; idx++)
				{
					if (idx > 0)
						appendStringInfoChar(&layerbuf, ',');
					appendStringInfo(&layerbuf, "%d", hidden_layers[idx]);
				}
				appendStringInfoChar(&layerbuf, ']');
				hidden_layers_json = layerbuf.data;
			}
			else
			{
				hidden_layers_json = pstrdup("[]");
			}

			appendStringInfo(&hyperbuf,
							 "{\"hidden_layers\":%s,\"activation\":\"%s\","
							 "\"learning_rate\":%.6f,\"epochs\":%d}",
							 hidden_layers_json,
							 activation,
							 learning_rate,
							 epochs);
			params_jsonb = DatumGetJsonbP(DirectFunctionCall1(
															  jsonb_in, CStringGetTextDatum(hyperbuf.data)));

			/* Build metrics JSON */
			initStringInfo(&metricsbuf);
			appendStringInfo(&metricsbuf,
							 "{\"algorithm\":\"neural_network\","
							 "\"storage\":\"cpu\","
							 "\"n_inputs\":%d,"
							 "\"n_outputs\":%d,"
							 "\"n_layers\":%d,"
							 "\"n_samples\":%d,"
							 "\"final_loss\":%.6f}",
							 n_inputs,
							 n_outputs,
							 net->n_layers,
							 n_samples,
							 loss);
			metrics_jsonb = DatumGetJsonbP(DirectFunctionCall1(
															   jsonb_in, CStringGetTextDatum(metricsbuf.data)));

			/* Register in catalog */
			memset(&spec, 0, sizeof(spec));
			spec.algorithm = "neural_network";
			spec.model_type = "regression";
			spec.training_table = table_name_str;
			spec.training_column = label_col_str;
			spec.parameters = params_jsonb;
			spec.metrics = metrics_jsonb;
			spec.model_data = serialized;
			spec.training_time_ms = -1;
			spec.num_samples = n_samples;
			spec.num_features = n_inputs;

			model_id = ml_catalog_register_model(&spec);


			nfree(sql.data);
			NDB_SPI_SESSION_END(train_nn_spi_session);
		}
		MemoryContextSwitchTo(oldcontext);
		MemoryContextDelete(callcontext);

		for (i = 0; i < n_samples; i++)
			nfree(X[i]);
		nfree(X);
		nfree(y);
		neural_network_free(net);
		nfree(hidden_layers);
		nfree(table_name_str);
		nfree(feature_col_str);
		nfree(label_col_str);
		nfree(activation);
		if (hidden_layers_json != NULL)
			nfree(hidden_layers_json);

		/*
		 * Note: serialized, params_jsonb, metrics_jsonb are owned by catalog
		 * now
		 */

		PG_RETURN_INT32(model_id);
	}
}

/*
 * Predict with neural network
 *
 * Loads trained neural network model and performs forward pass
 * to generate prediction.
 */
PG_FUNCTION_INFO_V1(predict_neural_network);
PG_FUNCTION_INFO_V1(evaluate_neural_network_by_model_id);

Datum
predict_neural_network(PG_FUNCTION_ARGS)
{
	Vector *features = NULL;
	int32		model_id;

	bytea *model_data = NULL;
	Jsonb *parameters = NULL;
	Jsonb *metrics = NULL;
	NeuralNetwork *net = NULL;

	float *input_features = NULL;
	float		result[1];
	int			i;

	/* Defensive: validate inputs */
	model_id = PG_GETARG_INT32(0);
	features = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(features);

	if (features == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("predict_neural_network: features cannot be NULL")));

	if (model_id <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: model_id must be positive")));

	if (features->dim <= 0 || features->dim > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: invalid feature dimension %d",
						features->dim)));

	/* Load model from catalog - ml_catalog_fetch_model_payload allocates in caller's context */
	if (!ml_catalog_fetch_model_payload(model_id, &model_data,
										&parameters, &metrics))
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: model %d not found", model_id)));
	}

	/* Defensive: validate model_data */
	if (model_data == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: model %d has no model data",
						model_id)));
	}

	/* Deserialize neural network from model_data with error handling */
	net = NULL;
	input_features = NULL;

	PG_TRY();
	{
		{
			uint8		training_backend = 0;

			net = neural_network_deserialize(model_data, &training_backend);
		}
		if (net == NULL)
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("predict_neural_network: failed to deserialize model %d",
							model_id)));
		}
	}
	PG_CATCH();
	{
		/* Cleanup on deserialization error */
		if (net != NULL)
			neural_network_free(net);
		PG_RE_THROW();
	}
	PG_END_TRY();

	/* Validate feature dimensions match network input */
	if (features->dim != net->n_inputs)
	{
		neural_network_free(net);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("predict_neural_network: feature dimension %d does not match model input dimension %d",
						features->dim, net->n_inputs)));
	}

	/* Copy features to input array with overflow check */
	{
		size_t		features_size = (size_t) features->dim * sizeof(float);

		if (features_size > MaxAllocSize)
		{
			neural_network_free(net);
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("predict_neural_network: feature array allocation exceeds maximum size")));
		}
		nalloc(input_features, float, features->dim);
		if (input_features == NULL)
		{
			neural_network_free(net);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("predict_neural_network: failed to allocate input features array")));
		}
	}
	for (i = 0; i < features->dim; i++)
	{
		if (!isfinite(features->data[i]))
		{
			nfree(input_features);
			neural_network_free(net);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("predict_neural_network: non-finite feature value at index %d",
							i)));
		}
		input_features[i] = features->data[i];
	}

	/* Perform forward pass with error handling */
	PG_TRY();
	{
		neural_network_forward(net, input_features, result);

		/* Validate result */
		if (!isfinite(result[0]))
		{
			neural_network_free(net);
			nfree(input_features);
			ereport(ERROR,
					(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
					 errmsg("predict_neural_network: non-finite prediction result")));
		}
	}
	PG_CATCH();
	{
		/* Cleanup on forward pass error */
		if (net != NULL)
			neural_network_free(net);
		if (input_features != NULL)
			nfree(input_features);
		PG_RE_THROW();
	}
	PG_END_TRY();

	neural_network_free(net);
	nfree(input_features);

	PG_RETURN_FLOAT8(result[0]);
}

/*
 * evaluate_neural_network_by_model_id
 *
 * Evaluates a neural network model on a dataset and returns performance metrics.
 * Arguments: int4 model_id, text table_name, text feature_col, text label_col
 * Returns: jsonb with metrics
 */
Datum
evaluate_neural_network_by_model_id(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text *table_name = NULL;
	text *feature_col = NULL;
	text *label_col = NULL;
	char *tbl_str = NULL;
	char *feat_str = NULL;
	char *targ_str = NULL;
	StringInfoData query;
	int			ret;
	int			nvec = 0;
	double		mse = 0.0;
	double		mae = 0.0;
	double		ss_tot = 0.0;
	double		ss_res = 0.0;
	double		y_mean = 0.0;
	double		r_squared;
	double		rmse;
	int			i;
	StringInfoData jsonbuf;
	Jsonb *result = NULL;
	MemoryContext oldcontext;

	NdbSpiSession *spi_session = NULL;

	/* Validate arguments */
	if (PG_NARGS() != 4)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_neural_network_by_model_id: 4 arguments are required")));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_neural_network_by_model_id: model_id is required")));

	model_id = PG_GETARG_INT32(0);

	if (PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_neural_network_by_model_id: table_name, feature_col, and label_col are required")));

	table_name = PG_GETARG_TEXT_PP(1);
	feature_col = PG_GETARG_TEXT_PP(2);
	label_col = PG_GETARG_TEXT_PP(3);

	tbl_str = text_to_cstring(table_name);
	feat_str = text_to_cstring(feature_col);
	targ_str = text_to_cstring(label_col);

	oldcontext = CurrentMemoryContext;

	/* Initialize and build query in caller's context BEFORE SPI_connect */
	initStringInfo(&query);
	appendStringInfo(&query,
					 "SELECT %s, %s FROM %s WHERE %s IS NOT NULL AND %s IS NOT NULL",
					 feat_str, targ_str, tbl_str, feat_str, targ_str);

	/* Connect to SPI */
	oldcontext = CurrentMemoryContext;

	NDB_SPI_SESSION_BEGIN(spi_session, oldcontext);

	ret = ndb_spi_execute(spi_session, query.data, true, 0);
	if (ret != SPI_OK_SELECT)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: evaluate_neural_network_by_model_id: query failed")));
	}

	nvec = SPI_processed;
	if (nvec < 2)
	{
		ndb_spi_stringinfo_free(spi_session, &query);
		NDB_SPI_SESSION_END(spi_session);
		nfree(tbl_str);
		nfree(feat_str);
		nfree(targ_str);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: evaluate_neural_network_by_model_id: need at least 2 samples, got %d",
						nvec)));
	}

	/* First pass: compute mean of y */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple;

		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		targ_datum;
			bool		targ_null;

			if (tupdesc == NULL)
			{
				continue;
			}

			/*
			 * Safe access for target - validate tupdesc has at least 2
			 * columns
			 */
			if (tupdesc->natts < 2)
			{
				continue;
			}
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);
			if (!targ_null)
				y_mean += DatumGetFloat8(targ_datum);
		}
	}
	y_mean /= nvec;

	/* Second pass: compute predictions and metrics */
	for (i = 0; i < nvec; i++)
	{
		HeapTuple	tuple;

		/* Safe access to SPI_tuptable - validate before access */
		if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
			i >= SPI_processed || SPI_tuptable->vals[i] == NULL)
		{
			continue;
		}
		tuple = SPI_tuptable->vals[i];
		{
			TupleDesc	tupdesc = SPI_tuptable->tupdesc;
			Datum		feat_datum;
			Datum		targ_datum;
			bool		feat_null;
			bool		targ_null;
			Vector	   *vec = NULL;
			double		y_true;
			double		y_pred;
			double		error;

			if (tupdesc == NULL)
			{
				continue;
			}

			feat_datum = SPI_getbinval(tuple, tupdesc, 1, &feat_null);

			/*
			 * Safe access for target - validate tupdesc has at least 2
			 * columns
			 */
			if (tupdesc->natts < 2)
			{
				continue;
			}
			targ_datum = SPI_getbinval(tuple, tupdesc, 2, &targ_null);

			if (feat_null || targ_null)
				continue;

			y_true = DatumGetFloat8(targ_datum);

			/* Extract features */
			vec = DatumGetVector(feat_datum);

			/* Make prediction using neural network model */
			y_pred = DatumGetFloat8(DirectFunctionCall2(predict_neural_network,
														Int32GetDatum(model_id),
														PointerGetDatum(vec)));

			/* Compute errors */
			error = y_true - y_pred;
			mse += error * error;
			mae += fabs(error);
			ss_res += error * error;
			ss_tot += (y_true - y_mean) * (y_true - y_mean);
		}
	}

	ndb_spi_stringinfo_free(spi_session, &query);
	NDB_SPI_SESSION_END(spi_session);

	mse /= nvec;
	mae /= nvec;
	rmse = sqrt(mse);

	/*
	 * Handle R² calculation - if ss_tot is zero (no variance in y), R² is
	 * undefined
	 */
	if (ss_tot == 0.0)
		r_squared = 0.0;		/* Convention: set to 0 when there's no
								 * variance to explain */
	else
		r_squared = 1.0 - (ss_res / ss_tot);

	/* Build result JSON */
	MemoryContextSwitchTo(oldcontext);
	initStringInfo(&jsonbuf);
	appendStringInfo(&jsonbuf,
					 "{\"mse\":%.6f,\"mae\":%.6f,\"rmse\":%.6f,\"r_squared\":%.6f,\"n_samples\":%d}",
					 mse, mae, rmse, r_squared, nvec);

	result = DatumGetJsonbP(DirectFunctionCall1(jsonb_in, CStringGetTextDatum(jsonbuf.data)));
	nfree(jsonbuf.data);

	nfree(tbl_str);
	nfree(feat_str);
	nfree(targ_str);

	PG_RETURN_JSONB_P(result);
}

#include "neurondb_gpu_model.h"
#include "ml_gpu_registry.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"

typedef struct NeuralNetworkGpuModelState
{
	bytea *model_blob;
	Jsonb *metrics;
	NeuralNetwork *network;
	int			n_inputs;
	int			n_outputs;
	int			n_hidden_layers;
	int *hidden_layer_sizes;
	char		activation_func[16];
	float		learning_rate;
	int			n_samples;
}			NeuralNetworkGpuModelState;

static bool
neural_network_gpu_train(MLGpuModel *model, const MLGpuTrainSpec *spec, char **errstr)
{
	NeuralNetworkGpuModelState *state = NULL;
	float **X = NULL;

	float *y = NULL;
	NeuralNetwork *net = NULL;
	int			nvec = 0;
	int			dim = 0;
	int			n_outputs = 1;

	int *hidden_layers = NULL;
	int			n_hidden = 1;
	char		activation[16] = "relu";
	float		learning_rate = 0.01f;
	int			epochs = 100;
	int			epoch,
				sample;
	float		loss;

	bytea *model_data = NULL;
	Jsonb *metrics = NULL;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;
	int			i;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || spec == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_train: invalid parameters");
		return false;
	}

	/* Ensure we're in TopMemoryContext for allocations in GPU context */
	MemoryContextSwitchTo(TopMemoryContext);

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
				if (strcmp(key, "hidden_layers") == 0 && v.type == jbvBinary)
				{
					JsonbIterator *arr_it = JsonbIteratorInit((JsonbContainer *) v.val.binary.data);
					JsonbValue	arr_v;
					int			arr_r;
					int			count = 0;

					int *temp_layers = NULL;
					int			capacity = 4;

					nalloc(temp_layers, int, capacity);
					while ((arr_r = JsonbIteratorNext(&arr_it, &arr_v, false)) != WJB_DONE)
					{
						if (arr_r == WJB_ELEM && arr_v.type == jbvNumeric)
						{
							if (count >= capacity)
							{
								capacity *= 2;
								temp_layers = (int *) repalloc(temp_layers, sizeof(int) * capacity);
							}
							temp_layers[count++] = DatumGetInt32(DirectFunctionCall1(numeric_int4,
																					 NumericGetDatum(arr_v.val.numeric)));
						}
					}
					if (count > 0)
					{
						nalloc(hidden_layers, int, count);
						memcpy(hidden_layers, temp_layers, sizeof(int) * count);
						n_hidden = count;
					}
					nfree(temp_layers);
				}
				else if (strcmp(key, "activation") == 0 && v.type == jbvString)
				{
					strncpy(activation, v.val.string.val, sizeof(activation) - 1);
					activation[sizeof(activation) - 1] = '\0';
				}
				else if (strcmp(key, "learning_rate") == 0 && v.type == jbvNumeric)
					learning_rate = (float) DatumGetFloat8(DirectFunctionCall1(numeric_float8,
																			   NumericGetDatum(v.val.numeric)));
				else if (strcmp(key, "epochs") == 0 && v.type == jbvNumeric)
					epochs = DatumGetInt32(DirectFunctionCall1(numeric_int4,
															   NumericGetDatum(v.val.numeric)));
				nfree(key);
			}
		}
	}

	if (n_hidden == 0 || hidden_layers == NULL)
	{
		n_hidden = 1;
		nalloc(hidden_layers, int, 1);
		hidden_layers[0] = 32;
	}

	if (strlen(activation) == 0)
	{
		strncpy(activation, "relu", sizeof(activation) - 1);
		activation[sizeof(activation) - 1] = '\0';
	}
	if (learning_rate <= 0.0f)
		learning_rate = 0.01f;
	if (epochs < 1)
		epochs = 100;

	/* Convert feature matrix to 2D array */
	if (spec->feature_matrix == NULL || spec->sample_count <= 0
		|| spec->feature_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_train: invalid feature matrix");
		return false;
	}

	nvec = spec->sample_count;
	dim = spec->feature_dim;

	/* Ensure we're in TopMemoryContext for allocations */
	MemoryContextSwitchTo(TopMemoryContext);

	/* Allocate training data */
	nalloc(X, float *, nvec);
	nalloc(y, float, nvec);

	for (i = 0; i < nvec; i++)
	{
		nalloc(X[i], float, dim);
		memcpy(X[i], &spec->feature_matrix[i * dim], sizeof(float) * dim);
		/* For regression, use first feature as target (simplified) */
		y[i] = X[i][0];
	}

	/* Initialize neural network */
	net = neural_network_init(dim, n_outputs, hidden_layers, n_hidden, activation, learning_rate);
	if (net == NULL)
	{
		for (i = 0; i < nvec; i++)
			nfree(X[i]);
		nfree(X);
		nfree(y);
		if (hidden_layers != NULL)
			nfree(hidden_layers);
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_train: failed to initialize network");
		return false;
	}

	/* Training loop */
	for (epoch = 0; epoch < epochs; epoch++)
	{
		loss = 0.0f;

		for (sample = 0; sample < nvec; sample++)
		{
			float		predicted[1];
			float		error;
			float		target[1];

			/* Forward pass */
			neural_network_forward(net, X[sample], predicted);

			/* Compute loss */
			error = y[sample] - predicted[0];
			loss += error * error;

			/* Backward pass */
			target[0] = y[sample];
			neural_network_backward(net, X[sample], target, predicted);

			/* Update weights */
			neural_network_update_weights(net, X[sample]);
		}

		loss /= nvec;
	}

	/* Serialize model */
	model_data = neural_network_serialize(net, 0); /* training_backend=0 for CPU */

	/* Build metrics using JSONB API directly to avoid DirectFunctionCall in GPU context */
	{
		JsonbParseState *state = NULL;
		JsonbValue	jkey;
		JsonbValue	jval;
		JsonbValue *final_value = NULL;
		MemoryContext oldcontext_jsonb = CurrentMemoryContext;
		Numeric		n_inputs_num, n_outputs_num, n_hidden_num, learning_rate_num, epochs_num, final_loss_num, n_samples_num;
		
		/* Switch to TopMemoryContext for JSONB construction */
		MemoryContextSwitchTo(TopMemoryContext);
		
		PG_TRY();
		{
			(void) pushJsonbValue(&state, WJB_BEGIN_OBJECT, NULL);
			
			/* Add storage */
			jkey.type = jbvString;
			jkey.val.string.val = "storage";
			jkey.val.string.len = strlen("storage");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvString;
			jval.val.string.val = "cpu";
			jval.val.string.len = strlen("cpu");
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add training_backend */
			jkey.val.string.val = "training_backend";
			jkey.val.string.len = strlen("training_backend");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvNumeric;
			jval.val.numeric = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(0)));
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add n_inputs */
			jkey.val.string.val = "n_inputs";
			jkey.val.string.len = strlen("n_inputs");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_inputs_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(dim)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_inputs_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add n_outputs */
			jkey.val.string.val = "n_outputs";
			jkey.val.string.len = strlen("n_outputs");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_outputs_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(n_outputs)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_outputs_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add n_hidden_layers */
			jkey.val.string.val = "n_hidden_layers";
			jkey.val.string.len = strlen("n_hidden_layers");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_hidden_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(n_hidden)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_hidden_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add activation */
			jkey.val.string.val = "activation";
			jkey.val.string.len = strlen("activation");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			jval.type = jbvString;
			jval.val.string.val = activation;
			jval.val.string.len = strlen(activation);
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add learning_rate */
			jkey.val.string.val = "learning_rate";
			jkey.val.string.len = strlen("learning_rate");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			learning_rate_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum((double) learning_rate)));
			jval.type = jbvNumeric;
			jval.val.numeric = learning_rate_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add epochs */
			jkey.val.string.val = "epochs";
			jkey.val.string.len = strlen("epochs");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			epochs_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(epochs)));
			jval.type = jbvNumeric;
			jval.val.numeric = epochs_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add final_loss */
			jkey.val.string.val = "final_loss";
			jkey.val.string.len = strlen("final_loss");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			final_loss_num = DatumGetNumeric(DirectFunctionCall1(float8_numeric, Float8GetDatum((double) loss)));
			jval.type = jbvNumeric;
			jval.val.numeric = final_loss_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			/* Add n_samples */
			jkey.val.string.val = "n_samples";
			jkey.val.string.len = strlen("n_samples");
			(void) pushJsonbValue(&state, WJB_KEY, &jkey);
			n_samples_num = DatumGetNumeric(DirectFunctionCall1(int4_numeric, Int32GetDatum(nvec)));
			jval.type = jbvNumeric;
			jval.val.numeric = n_samples_num;
			(void) pushJsonbValue(&state, WJB_VALUE, &jval);
			
			final_value = pushJsonbValue(&state, WJB_END_OBJECT, NULL);
			if (final_value == NULL)
			{
				MemoryContextSwitchTo(oldcontext_jsonb);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: neural_network_gpu_train: pushJsonbValue(WJB_END_OBJECT) returned NULL")));
			}
			
			metrics = JsonbValueToJsonb(final_value);
		}
		PG_CATCH();
		{
			MemoryContextSwitchTo(oldcontext_jsonb);
			PG_RE_THROW();
		}
		PG_END_TRY();
		
		MemoryContextSwitchTo(oldcontext_jsonb);
	}

	nalloc(state, NeuralNetworkGpuModelState, 1);
	state->model_blob = model_data;
	state->metrics = metrics;
	state->network = net;
	state->n_inputs = dim;
	state->n_outputs = n_outputs;
	state->n_hidden_layers = n_hidden;
	state->hidden_layer_sizes = hidden_layers;
	strncpy(state->activation_func, activation, sizeof(state->activation_func) - 1);
	state->learning_rate = learning_rate;
	state->n_samples = nvec;

	if (model->backend_state != NULL)
		nfree(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	/* Cleanup temp data */
	for (i = 0; i < nvec; i++)
		nfree(X[i]);
	nfree(X);
	nfree(y);

	return true;
}

static bool
neural_network_gpu_predict(const MLGpuModel *model, const float *input, int input_dim,
						   float *output, int output_dim, char **errstr)
{
	const		NeuralNetworkGpuModelState *state;

	NeuralNetwork *net = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (output != NULL && output_dim > 0)
		output[0] = 0.0f;
	if (model == NULL || input == NULL || output == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_predict: invalid parameters");
		return false;
	}
	if (output_dim <= 0)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_predict: invalid output dimension");
		return false;
	}
	if (!model->gpu_ready || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_predict: model not ready");
		return false;
	}

	state = (const NeuralNetworkGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_predict: model blob is NULL");
		return false;
	}

	if (input_dim != state->n_inputs)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_predict: dimension mismatch");
		return false;
	}

	/* Deserialize network if not already loaded */
	if (state->network == NULL)
	{
		{
			uint8		training_backend = 0;

			net = neural_network_deserialize(state->model_blob, &training_backend);
		}
		if (net == NULL)
		{
			if (errstr != NULL)
				*errstr = pstrdup("neural_network_gpu_predict: failed to deserialize");
			return false;
		}
		((NeuralNetworkGpuModelState *) state)->network = net;
	}
	else
	{
		net = state->network;
	}

	/* Forward pass */
	neural_network_forward(net, (float *) input, output);

	return true;
}

static bool
neural_network_gpu_evaluate(const MLGpuModel *model, const MLGpuEvalSpec *spec,
							MLGpuMetrics *out, char **errstr)
{
	const		NeuralNetworkGpuModelState *state;
	Jsonb	   *metrics_json = NULL;
	StringInfoData buf;

	if (errstr != NULL)
		*errstr = NULL;
	if (out != NULL)
		out->payload = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_evaluate: invalid model");
		return false;
	}

	state = (const NeuralNetworkGpuModelState *) model->backend_state;

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "{\"algorithm\":\"neural_network\",\"storage\":\"cpu\","
					 "\"n_inputs\":%d,\"n_outputs\":%d,\"n_hidden_layers\":%d,\"activation\":\"%s\",\"learning_rate\":%.6f,\"n_samples\":%d}",
					 state->n_inputs > 0 ? state->n_inputs : 0,
					 state->n_outputs > 0 ? state->n_outputs : 1,
					 state->n_hidden_layers > 0 ? state->n_hidden_layers : 1,
					 state->activation_func[0] ? state->activation_func : "relu",
					 state->learning_rate > 0.0f ? state->learning_rate : 0.01f,
					 state->n_samples > 0 ? state->n_samples : 0);

	metrics_json = DatumGetJsonbP(DirectFunctionCall1(jsonb_in,
													  CStringGetTextDatum(buf.data)));
	nfree(buf.data);

	if (out != NULL)
		out->payload = metrics_json;

	return true;
}

static bool
neural_network_gpu_serialize(const MLGpuModel *model, bytea * *payload_out,
							 Jsonb * *metadata_out, char **errstr)
{
	const		NeuralNetworkGpuModelState *state;
	bytea	   *payload_copy = NULL;
	int			payload_size;
	char	   *payload_copy_raw = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (payload_out != NULL)
		*payload_out = NULL;
	if (metadata_out != NULL)
		*metadata_out = NULL;
	if (model == NULL || model->backend_state == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_serialize: invalid model");
		return false;
	}

	state = (const NeuralNetworkGpuModelState *) model->backend_state;
	if (state->model_blob == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_serialize: model blob is NULL");
		return false;
	}

	payload_size = VARSIZE(state->model_blob);
	nalloc(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, state->model_blob, payload_size);

	if (payload_out != NULL)
		*payload_out = payload_copy;
	else
		nfree(payload_copy);

	if (metadata_out != NULL && state->metrics != NULL)
		*metadata_out = (Jsonb *) PG_DETOAST_DATUM_COPY(
														PointerGetDatum(state->metrics));

	return true;
}

static bool
neural_network_gpu_deserialize(MLGpuModel *model, const bytea * payload,
							   const Jsonb * metadata, char **errstr)
{
	NeuralNetworkGpuModelState *state = NULL;
	bytea	   *payload_copy = NULL;
	int			payload_size;
	NeuralNetwork *net = NULL;
	JsonbIterator *it = NULL;
	JsonbValue	v;
	int			r;
	char	   *payload_copy_raw = NULL;

	if (errstr != NULL)
		*errstr = NULL;
	if (model == NULL || payload == NULL)
	{
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_deserialize: invalid parameters");
		return false;
	}

	payload_size = VARSIZE(payload);
	nalloc(payload_copy_raw, char, payload_size);
	payload_copy = (bytea *) payload_copy_raw;
	memcpy(payload_copy, payload, payload_size);

	{
		uint8		training_backend = 0;

		net = neural_network_deserialize(payload_copy, &training_backend);
	}
	if (net == NULL)
	{
		nfree(payload_copy);
		if (errstr != NULL)
			*errstr = pstrdup("neural_network_gpu_deserialize: failed to deserialize");
		return false;
	}

	nalloc(state, NeuralNetworkGpuModelState, 1);
	state->model_blob = payload_copy;
	state->network = net;
	state->n_inputs = net->n_inputs;
	state->n_outputs = net->n_outputs;
	state->n_hidden_layers = net->n_layers - 1;
	state->learning_rate = net->learning_rate;
	strncpy(state->activation_func, net->activation_func, sizeof(state->activation_func) - 1);
	state->n_samples = 0;

	if (net->n_layers > 1)
	{
		int *hidden_layer_sizes_tmp = NULL;
		nalloc(hidden_layer_sizes_tmp, int, state->n_hidden_layers);
		for (int i = 0; i < state->n_hidden_layers; i++)
			hidden_layer_sizes_tmp[i] = net->layers[i].n_outputs;
		state->hidden_layer_sizes = hidden_layer_sizes_tmp;
	}

	if (metadata != NULL)
	{
		int			metadata_size = VARSIZE(metadata);

		Jsonb *metadata_copy = NULL;
		char *metadata_copy_raw = NULL;
		nalloc(metadata_copy_raw, char, metadata_size);
		metadata_copy = (Jsonb *) metadata_copy_raw;

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
				nfree(key);
			}
		}
	}
	else
	{
		state->metrics = NULL;
	}

	if (model->backend_state != NULL)
		nfree(model->backend_state);

	model->backend_state = state;
	model->gpu_ready = true;
	model->is_gpu_resident = false;

	return true;
}

static void
neural_network_gpu_destroy(MLGpuModel *model)
{
	NeuralNetworkGpuModelState *state = NULL;

	if (model == NULL)
		return;

	if (model->backend_state != NULL)
	{
		state = (NeuralNetworkGpuModelState *) model->backend_state;
		if (state->model_blob != NULL)
			nfree(state->model_blob);
		if (state->metrics != NULL)
			nfree(state->metrics);
		if (state->network != NULL)
			neural_network_free(state->network);
		if (state->hidden_layer_sizes != NULL)
			nfree(state->hidden_layer_sizes);
		nfree(state);
		model->backend_state = NULL;
	}

	model->gpu_ready = false;
	model->is_gpu_resident = false;
}

static const MLGpuModelOps neural_network_gpu_model_ops = {
	.algorithm = "neural_network",
	.train = neural_network_gpu_train,
	.predict = neural_network_gpu_predict,
	.evaluate = neural_network_gpu_evaluate,
	.serialize = neural_network_gpu_serialize,
	.deserialize = neural_network_gpu_deserialize,
	.destroy = neural_network_gpu_destroy,
};

void
neurondb_gpu_register_neural_network_model(void)
{
	static bool registered = false;

	if (registered)
		return;
	ndb_gpu_register_model_ops(&neural_network_gpu_model_ops);
	registered = true;
}
