/*-------------------------------------------------------------------------
 *
 * gpu_clustering.c
 *	GPU-accelerated clustering algorithms
 *
 * Implements KMeans clustering helpers. DBSCAN currently falls back to
 * the CPU implementation until dedicated GPU kernels are available.
 *
 *-------------------------------------------------------------------------*/

#include "postgres.h"
#include "fmgr.h"

#include "neurondb_config.h"
#include "neurondb_gpu.h"
#include "neurondb_gpu_backend.h"
#include "neurondb_constants.h"

/*
 * GPU KMeans clustering
 */
void
neurondb_gpu_kmeans(const float *vectors,
					int num_vectors,
					int dim,
					int k,
					int max_iters,
					float *centroids,
					int *assignments)
{
	const ndb_gpu_backend *backend;
	int			iter;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		return;

	if (!neurondb_gpu_is_available())
		return;

	backend = ndb_gpu_get_active_backend();
	if (backend == NULL)
		return;

	if (backend->launch_kmeans_assign == NULL
		|| backend->launch_kmeans_update == NULL)
	{
		elog(DEBUG1,
			 "neurondb: active GPU backend lacks k-means support; "
			 "falling back to CPU");
		return;
	}

	for (iter = 0; iter < max_iters; iter++)
	{
		if (backend->launch_kmeans_assign(vectors,
										  centroids,
										  assignments,
										  num_vectors,
										  dim,
										  k,
										  NULL)
			!= 0)
		{
			elog(DEBUG1,
				 "neurondb: GPU k-means assign failed (iter "
				 "%d); falling back",
				 iter);
			return;
		}

		if (backend->launch_kmeans_update(vectors,
										  assignments,
										  centroids,
										  num_vectors,
										  dim,
										  k,
										  NULL)
			!= 0)
		{
			elog(DEBUG1,
				 "neurondb: GPU k-means update failed (iter "
				 "%d); falling back",
				 iter);
			return;
		}
	}
}

/*
 * GPU DBSCAN clustering
 * Currently unimplemented; always use CPU path.
 */
void
neurondb_gpu_dbscan(const float *vectors,
					int num_vectors,
					int dim,
					float eps,
					int min_pts,
					int *labels)
{
	(void) vectors;
	(void) num_vectors;
	(void) dim;
	(void) eps;
	(void) min_pts;
	(void) labels;

	/* CPU mode: never run GPU code */
	if (NDB_COMPUTE_MODE_IS_CPU())
		return;

	if (!neurondb_gpu_is_available())
		return;

	elog(DEBUG1,
		 "neurondb: GPU DBSCAN not implemented; using CPU fallback");
}
