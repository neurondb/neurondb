/*-------------------------------------------------------------------------
 *
 * gpu_model_registry.c
 *    Model operator registry.
 *
 * This module allows individual algorithms to register native implementations.
 * The registry is consulted by unified ML entry points for training and
 * prediction routing.
 *
 * Copyright (c) 2024-2026, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/gpu/common/gpu_model_registry.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "neurondb_gpu_model.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"

typedef struct MLGpuModelEntry
{
	char		algorithm[64];	/* Algorithm name as key */
	const MLGpuModelOps *ops;
}			MLGpuModelEntry;

static HTAB *gpu_model_registry = NULL;

static void
ndb_gpu_init_model_registry(void)
{
	HASHCTL		ctl;

	if (gpu_model_registry != NULL)
		return;

	MemSet(&ctl, 0, sizeof(HASHCTL));
	ctl.keysize = 64;			/* Maximum algorithm name length */
	ctl.entrysize = sizeof(MLGpuModelEntry);
	ctl.hcxt = TopMemoryContext;
	gpu_model_registry = hash_create("neurondb GPU model registry",
									 16,
									 &ctl,
									 HASH_ELEM | HASH_STRINGS | HASH_CONTEXT);
}

bool
ndb_gpu_register_model_ops(const MLGpuModelOps *ops)
{
	MLGpuModelEntry *entry = NULL;
	bool		found;
	char		key[64];

	if (ops == NULL || ops->algorithm == NULL)
		return false;

	ndb_gpu_init_model_registry();

	strlcpy(key, ops->algorithm, sizeof(key));

	entry = (MLGpuModelEntry *) hash_search(
											gpu_model_registry, key, HASH_ENTER, &found);
	if (entry == NULL)
		return false;
	entry->ops = ops;

	return !found;
}

const MLGpuModelOps *
ndb_gpu_lookup_model_ops(const char *algorithm)
{
	MLGpuModelEntry *entry = NULL;

	if (algorithm == NULL || gpu_model_registry == NULL)
	{
		return NULL;
	}

	entry = (MLGpuModelEntry *) hash_search(
											gpu_model_registry, algorithm, HASH_FIND, NULL);


	if (entry == NULL)
		return NULL;
	return entry->ops;
}

void
ndb_gpu_clear_model_registry(void)
{
	if (gpu_model_registry != NULL)
	{
		hash_destroy(gpu_model_registry);
		gpu_model_registry = NULL;
	}
}
