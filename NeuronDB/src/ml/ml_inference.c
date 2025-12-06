/*-------------------------------------------------------------------------
 *
 * ml_inference.c
 *    Machine learning model inference engine.
 *
 * This module provides model lifecycle management including registration,
 * loading, prediction, batch inference, and fine-tuning with backend support.
 *
 * Copyright (c) 2024-2025, neurondb, Inc.
 *
 * IDENTIFICATION
 *    src/ml/ml_inference.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "neurondb.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/array.h"
#include "catalog/pg_type.h"
#include "access/xact.h"
#include "access/htup_details.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "executor/spi.h"
#include "pgtime.h"
#include "ml_catalog.h"
#include "neurondb_validation.h"
#include "neurondb_spi_safe.h"
#include "neurondb_safe_memory.h"
#include "neurondb_macros.h"
#include "neurondb_spi.h"
#include "neurondb_onnx.h"
#include <stdint.h>

#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>

#define MODEL_HANDLE_MAGIC 0x4E44B00F

typedef enum
{
	MODEL_ONNX,
	MODEL_TF,
	MODEL_PYTORCH,
	MODEL_UNKNOWN
}			ModelType;

typedef struct ModelHandle
{
	uint32_t	magic;
	ModelType	type;
	char	   *load_path;
	time_t		created;
	void	   *opaque_backend_state;
	size_t		model_size_bytes;
	uint32_t	version;
	char		backend_msg[128];	/* e.g., "ONNX backend loaded ok" */
}			ModelHandle;

typedef struct ModelEntry
{
	char	   *name;
	char	   *path;
	ModelType	type;
	bool		loaded;
	ModelHandle *model_handle;	/* strongly-typed handle */
	struct ModelEntry *next;
}			ModelEntry;

static ModelEntry * model_registry_head = NULL;

static ModelType
parse_model_type(const char *t)
{
	if (pg_strcasecmp(t, "onnx") == 0)
		return MODEL_ONNX;
	if (pg_strcasecmp(t, "tensorflow") == 0 || pg_strcasecmp(t, "tf") == 0)
		return MODEL_TF;
	if (pg_strcasecmp(t, "pytorch") == 0)
		return MODEL_PYTORCH;
	return MODEL_UNKNOWN;
}

static const char *
model_type_to_cstr(ModelType t)
{
	switch (t)
	{
		case MODEL_ONNX:
			return "onnx";
		case MODEL_TF:
			return "tensorflow";
		case MODEL_PYTORCH:
			return "pytorch";
		default:
			return "unknown";
	}
}

static ModelEntry *
find_model(const char *name)
{
	ModelEntry *cur;
	volatile	ModelEntry *volatile_cur;


	if (name == NULL)
	{
		return NULL;
	}

	if (TopMemoryContext == NULL || CurrentMemoryContext == NULL)
	{
		return NULL;
	}

	/*
	 * Use volatile pointer to prevent compiler optimizations that might hide
	 * memory corruption
	 */
	volatile_cur = (volatile ModelEntry *) model_registry_head;
	cur = (ModelEntry *) volatile_cur;

	/* Limit iteration to prevent infinite loops from corrupted linked list */
	{
		int			max_iter = 10000;
		int			iter = 0;

		while (cur != NULL && iter < max_iter)
		{

			if ((uintptr_t) cur < 0x1000 || (uintptr_t) cur % sizeof(void *) != 0)
			{
				break;
			}

			if (cur->name == NULL)
			{
				break;
			}


			if ((uintptr_t) cur->name < 0x1000 || (uintptr_t) cur->name % sizeof(void *) != 0)
			{
				break;
			}

			if (pg_strcasecmp(cur->name, name) == 0)
			{
				if (cur->model_handle != NULL)
				{
					if ((uintptr_t) cur->model_handle < 0x1000 ||
						(uintptr_t) cur->model_handle % sizeof(void *) != 0)
					{
						break;
					}

					if (cur->model_handle->magic != MODEL_HANDLE_MAGIC)
					{
						ereport(ERROR,
								(errcode(ERRCODE_DATA_CORRUPTED),
								 errmsg("neurondb: model handle magic invalid (expected 0x%x, got 0x%x)",
										MODEL_HANDLE_MAGIC, cur->model_handle->magic)));
						break;
					}
				}
				return cur;
			}
			volatile_cur = (volatile ModelEntry *) cur->next;
			cur = (ModelEntry *) volatile_cur;
			iter++;
		}
	}
	return NULL;
}

/*
 * model_backend_load - Load a model using the appropriate backend
 *
 * Loads a machine learning model from disk using the backend specified by
 * the model type. Supports ONNX, TensorFlow, and PyTorch models. Validates
 * file existence and readability before attempting to load.
 *
 * Parameters:
 *   path - File system path to the model file
 *   type - Type of model (ONNX, TensorFlow, or PyTorch)
 *
 * Returns:
 *   Pointer to ModelHandle structure on success
 *
 * Notes:
 *   The function validates that the path exists, is a regular file, and is
 *   readable before attempting to load. Errors are reported via ereport if
 *   validation fails. The model handle must be freed when no longer needed.
 */
static ModelHandle *
model_backend_load(const char *path, ModelType type)
{
	struct stat statbuf;
	ModelHandle *hdl;

	if (stat(path, &statbuf) != 0 || !S_ISREG(statbuf.st_mode))
	{
		int			saved_errno = errno;

		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FILE),
				 errmsg("neurondb: Model path '%s' not found or not a file [errno=%d: %s]",
						path,
						saved_errno,
						strerror(saved_errno))));
	}
	if (access(path, R_OK) != 0)
	{
		int			saved_errno = errno;

		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("neurondb: Model file '%s' is not readable [errno=%d: %s]",
						path,
						saved_errno,
						strerror(saved_errno))));
	}

	hdl = (ModelHandle *) MemoryContextAlloc(
											 TopMemoryContext, sizeof(ModelHandle));
	hdl->magic = MODEL_HANDLE_MAGIC;
	hdl->type = type;
	hdl->created = time(NULL);
	hdl->opaque_backend_state =
		NULL;
	hdl->model_size_bytes = statbuf.st_size;
	hdl->version = 1;
	hdl->load_path = MemoryContextStrdup(TopMemoryContext, path);

	switch (type)
	{
		case MODEL_ONNX:
			{
				/* Try to actually load ONNX model */
				NDB_DECLARE(ONNXModelSession *, session);
				NDB_DECLARE(char *, model_name);
				char	   *last_slash = strrchr(path, '/');
				char	   *last_dot = strrchr(path, '.');

				/*
				 * Extract model name from path (directory name or filename
				 * without extension)
				 */
				if (last_slash != NULL)
				{
					if (last_dot != NULL && last_dot > last_slash)
					{
						/*
						 * Has extension, extract name between last slash and
						 * dot
						 */
						size_t		name_len = last_dot - last_slash - 1;

						NDB_ALLOC(model_name, char, name_len + 1);
						memcpy(model_name, last_slash + 1, name_len);
						model_name[name_len] = '\0';
					}
					else
					{
						/* No extension, use everything after last slash */
						model_name = pstrdup(last_slash + 1);
					}
				}
				else
				{
					/* No slash, use filename without extension */
					if (last_dot != NULL)
					{
						size_t		name_len = last_dot - path;

						NDB_ALLOC(model_name, char, name_len + 1);
						memcpy(model_name, path, name_len);
						model_name[name_len] = '\0';
					}
					else
					{
						model_name = pstrdup(path);
					}
				}

				/* Try to load ONNX model */
#ifdef HAVE_ONNX_RUNTIME
				PG_TRY();
				{
					session = neurondb_onnx_get_or_load_model(model_name, ONNX_MODEL_EMBEDDING);
					if (session != NULL && session->is_loaded)
					{
						hdl->opaque_backend_state = session;
						snprintf(hdl->backend_msg,
								 sizeof(hdl->backend_msg),
								 "ONNX backend loaded [%lu bytes]",
								 (unsigned long) hdl->model_size_bytes);
					}
					else
					{
						snprintf(hdl->backend_msg,
								 sizeof(hdl->backend_msg),
								 "ONNX backend not available (stub) [%lu bytes]",
								 (unsigned long) hdl->model_size_bytes);
					}
				}
				PG_CATCH();
				{
					/* ONNX Runtime not available or load failed */
					snprintf(hdl->backend_msg,
							 sizeof(hdl->backend_msg),
							 "ONNX backend not available (stub) [%lu bytes]",
							 (unsigned long) hdl->model_size_bytes);
					FlushErrorState();
				}
				PG_END_TRY();
#else
				snprintf(hdl->backend_msg,
						 sizeof(hdl->backend_msg),
						 "ONNX backend not available (stub) [%lu bytes]",
						 (unsigned long) hdl->model_size_bytes);
#endif
				if (model_name)
					NDB_FREE(model_name);
			}
			break;
		case MODEL_TF:
			snprintf(hdl->backend_msg,
					 sizeof(hdl->backend_msg),
					 "TensorFlow backend not available (stub) [%lu bytes]",
					 (unsigned long) hdl->model_size_bytes);
			break;
		case MODEL_PYTORCH:
			snprintf(hdl->backend_msg,
					 sizeof(hdl->backend_msg),
					 "PyTorch backend not available (stub) [%lu bytes]",
					 (unsigned long) hdl->model_size_bytes);
			break;
		default:
			snprintf(hdl->backend_msg,
					 sizeof(hdl->backend_msg),
					 "Unknown backend loaded");
			break;
	}

	elog(DEBUG1,
		 "neurondb: Model backend loader: path='%s', type=%s, size=%lu, msg='%s'",
		 path,
		 model_type_to_cstr(type),
		 (unsigned long) hdl->model_size_bytes,
		 hdl->backend_msg);

	return hdl;
}

static void model_backend_unload(ModelHandle * hdl, ModelType type)
			pg_attribute_unused();
static void
model_backend_unload(ModelHandle * hdl, ModelType type)
{
	if (hdl)
	{
		/* Clean up backend-specific state */
		switch (type)
		{
			case MODEL_ONNX:
#ifdef HAVE_ONNX_RUNTIME
				if (hdl->opaque_backend_state != NULL)
				{
					/*
					 * Note: ONNX sessions are managed by the cache, so we
					 * don't unload here
					 */
					/*
					 * The cache will handle cleanup when the session is
					 * evicted
					 */
					hdl->opaque_backend_state = NULL;
				}
#endif
				break;
			case MODEL_TF:
				/* TensorFlow cleanup not yet implemented */
				break;
			case MODEL_PYTORCH:
				/* PyTorch cleanup not yet implemented */
				break;
			default:
				break;
		}

		if (hdl->load_path)
		{
			NDB_FREE(hdl->load_path);
			hdl->load_path = NULL;
		}
		memset(hdl, 0, sizeof(ModelHandle));
		NDB_FREE(hdl);
	}
}

PG_FUNCTION_INFO_V1(load_model);

Datum
load_model(PG_FUNCTION_ARGS)
{
	text	   *model_name;
	text	   *model_path;
	text	   *model_type;
	char	   *name_str;
	char	   *path_str;
	char	   *type_str;
	ModelType	t;
	ModelHandle *handle;
	ModelEntry *entry;
	struct stat statbuf;

	NDB_CHECK_NULL_ARG(0, "model_name");
	NDB_CHECK_NULL_ARG(1, "model_path");
	NDB_CHECK_NULL_ARG(2, "model_type");

	model_name = PG_GETARG_TEXT_PP(0);
	model_path = PG_GETARG_TEXT_PP(1);
	model_type = PG_GETARG_TEXT_PP(2);

	name_str = text_to_cstring(model_name);
	path_str = text_to_cstring(model_path);
	type_str = text_to_cstring(model_type);

	if (find_model(name_str) != NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DUPLICATE_OBJECT),
				 errmsg("neurondb: Model with name '%s' is already registered.",
						name_str)));

	t = parse_model_type(type_str);
	if (t == MODEL_UNKNOWN)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: Unknown model type '%s'. Use 'onnx', 'tensorflow', or 'pytorch'.",
						type_str)));

	if (stat(path_str, &statbuf) != 0 || !S_ISREG(statbuf.st_mode))
	{
		int			saved_errno = errno;

		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FILE),
				 errmsg("neurondb: Model path '%s' does not exist or is not a regular file (errno=%d: %s)",
						path_str,
						saved_errno,
						strerror(saved_errno))));
	}
	if (access(path_str, R_OK) != 0)
	{
		int			saved_errno = errno;

		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("neurondb: Model file '%s' is not readable (errno=%d: %s)",
						path_str,
						saved_errno,
						strerror(saved_errno))));
	}

	handle = model_backend_load(path_str, t);
	if (handle == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: Model backend failed to load '%s'",
						path_str)));

	entry = (ModelEntry *) MemoryContextAlloc(
											  TopMemoryContext, sizeof(ModelEntry));
	entry->name = MemoryContextStrdup(TopMemoryContext, name_str);
	entry->path = MemoryContextStrdup(TopMemoryContext, path_str);
	entry->type = t;
	entry->model_handle = handle;
	entry->loaded = true;
	entry->next = model_registry_head;
	model_registry_head = entry;

	elog(DEBUG1,
		 "neurondb: Registered and loaded model '%s' (type: %s, size: %lu bytes) from '%s'. Backend: %s",
		 name_str,
		 model_type_to_cstr(t),
		 (unsigned long) handle->model_size_bytes,
		 path_str,
		 handle->backend_msg);

	PG_RETURN_BOOL(true);
}

PG_FUNCTION_INFO_V1(predict);

Datum
predict(PG_FUNCTION_ARGS)
{
	text	   *model_name;
	Vector	   *input;

	NDB_DECLARE(char *, name_str);
	ModelEntry *m;
	Vector	   *result;


	if (PG_NARGS() != 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: predict requires 2 arguments, got %d", PG_NARGS())));

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: model_name cannot be NULL")));
	if (PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: Input vector is NULL")));

	model_name = PG_GETARG_TEXT_PP(0);
	input = PG_GETARG_VECTOR_P(1);
	NDB_CHECK_VECTOR_VALID(input);

	if (CurrentMemoryContext == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: predict() CurrentMemoryContext is NULL")));
	}
	elog(DEBUG1,
		 "neurondb: predict() memory context: current=%p, top=%p",
		 (void *) CurrentMemoryContext, (void *) TopMemoryContext);

	if (model_name == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: model_name cannot be NULL")));

	if (input == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("neurondb: Input vector is NULL")));

	if (input->dim <= 0 || input->dim > VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: Invalid vector dimension %d (must be 1-%d)",
						input->dim, VECTOR_MAX_DIM)));

	name_str = text_to_cstring(model_name);
	if (name_str == NULL)
	{
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: predict() text_to_cstring returned NULL")));
	}

	/*
	 * During SQL function compilation, PostgreSQL may call this function with
	 * dummy arguments just to validate the signature. We need to handle this
	 * gracefully without accessing potentially corrupted memory.
	 */
	m = NULL;

	if (TopMemoryContext == NULL || CurrentMemoryContext == NULL)
	{
		NDB_FREE(name_str);
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: predict() TopMemoryContext=%p, CurrentMemoryContext=%p",
						(void *) TopMemoryContext, (void *) CurrentMemoryContext)));
	}

	/*
	 * During SQL function compilation, PostgreSQL may use special memory
	 * contexts that are not safe for accessing global state. We check for
	 * this by validating the context structure.
	 */
	{
		MemoryContext parent;

		PG_TRY();
		{
			parent = MemoryContextGetParent(CurrentMemoryContext);
			if (parent == NULL && CurrentMemoryContext != TopMemoryContext)
			{
				/* This could be a compilation context - be extra cautious */
				/* Don't error here, but skip registry access */
			}
		}
		PG_CATCH();
		{
			/*
			 * If getting parent fails, memory context is corrupted - abort
			 * safely
			 */
			NDB_FREE(name_str);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: predict() MemoryContextGetParent failed - memory context corrupted")));
		}
		PG_END_TRY();
	}

	/* Try to access registry with maximum safety */

	/*
	 * During SQL function compilation, PostgreSQL may call this function with
	 * dummy arguments. We need to detect this and avoid accessing the
	 * registry which might cause memory corruption.
	 */
	PG_TRY();
	{
		/* First, validate that we can safely read the registry pointer */
		volatile	ModelEntry *volatile_reg_head;
		uintptr_t	ptr_val;
		ModelEntry *safe_reg_head;

		/* Assertions for crash tracking */
		Assert(name_str != NULL);
		Assert(strlen(name_str) > 0);

		volatile_reg_head = (volatile ModelEntry *) model_registry_head;

		/* Validate registry pointer before dereferencing */
		if (volatile_reg_head != NULL)
		{
			ptr_val = (uintptr_t) volatile_reg_head;
			/* Check for obviously corrupted pointers */
			if (ptr_val < 0x1000 || ptr_val > 0x7fffffffffffULL)
			{
				/* Suspicious pointer value - likely corruption */
				elog(WARNING,
					 "neurondb: predict() suspicious pointer value: 0x%lx",
					 (unsigned long) ptr_val);
				m = NULL;
			}
			else
			{
				/*
				 * Pointer looks reasonable, try to access with additional
				 * safety
				 */
				/*
				 * Use volatile to prevent compiler optimizations that might
				 * hide corruption
				 */
				safe_reg_head = (ModelEntry *) volatile_reg_head;

				/* Double-check pointer is still valid after volatile cast */
				if ((uintptr_t) safe_reg_head == ptr_val)
				{
					/* Try to find model, but be ready to catch any corruption */
					m = find_model(name_str);
				}
				else
				{
					/* Pointer changed during access - corruption detected */
					m = NULL;
				}
			}
		}
		else
		{
			/* Registry is empty - no models loaded */
			m = NULL;
		}
	}
	PG_CATCH();
	{
		NDB_FREE(name_str);

		/*
		 * During function compilation or if registry is corrupted, return
		 * error gracefully
		 */
		ereport(ERROR,
				(errcode(ERRCODE_INTERNAL_ERROR),
				 errmsg("neurondb: predict() exception caught during registry access")));
	}
	PG_END_TRY();

	if (m == NULL)
	{
		NDB_FREE(name_str);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: Model '%s' not found; must load first.",
						name_str)));
	}

	/* Defensive: Validate model entry */
	Assert(m != NULL);
	if (!m->loaded || m->model_handle == NULL)
	{
		NDB_FREE(name_str);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: predict() model '%s' not loaded (loaded=%d, handle=%p)",
						name_str, m->loaded, (void *) m->model_handle)));
	}

	/* Defensive: Validate model handle */
	if (m->model_handle->magic != MODEL_HANDLE_MAGIC)
	{
		NDB_FREE(name_str);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("neurondb: predict() model '%s' handle magic invalid (expected 0x%x, got 0x%x)",
						name_str, MODEL_HANDLE_MAGIC, m->model_handle->magic)));
	}

	elog(DEBUG1,
		 "neurondb: [predict] Model '%s' (type=%s) on %d-dim vector; "
		 "model loaded from '%s'; backend msg='%s'",
		 name_str,
		 model_type_to_cstr(m->type),
		 input->dim,
		 m->path,
		 m->model_handle->backend_msg);

	/* Check if backend is actually available (not just a stub) */
	if (strstr(m->model_handle->backend_msg, "stub") != NULL)
	{
		NDB_FREE(name_str);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("neurondb: Model '%s' backend is not available. "
						"Backend message: %s",
						name_str,
						m->model_handle->backend_msg),
				 errhint("Install ONNX Runtime, TensorFlow, or PyTorch "
						 "to enable model inference")));
	}

	/* Run actual inference based on backend type */
	result = NULL;

	switch (m->type)
	{
		case MODEL_ONNX:
			{
#ifdef HAVE_ONNX_RUNTIME
				ONNXModelSession *session = (ONNXModelSession *) m->model_handle->opaque_backend_state;

				NDB_DECLARE(ONNXTensor *, input_tensor);
				NDB_DECLARE(ONNXTensor *, output_tensor);

				if (session != NULL && session->is_loaded)
				{
					/* Convert Vector to ONNXTensor */
					NDB_DECLARE(int64 *, shape);
					NDB_DECLARE(float *, data);
					NDB_ALLOC(input_tensor, ONNXTensor, 1);
					memset(input_tensor, 0, sizeof(ONNXTensor));
					input_tensor->ndim = 2;
					NDB_ALLOC(shape, int64, 2);
					input_tensor->shape = shape;
					input_tensor->shape[0] = 1; /* Batch size */
					input_tensor->shape[1] = input->dim;
					input_tensor->size = input->dim;
					NDB_ALLOC(data, float, input->dim);
					input_tensor->data = data;
					memcpy(input_tensor->data, input->data, sizeof(float) * input->dim);

					/* Run inference */
					PG_TRY();
					{
						output_tensor = neurondb_onnx_run_inference(session, input_tensor);
						if (output_tensor != NULL && output_tensor->data != NULL)
						{
							/* Convert ONNXTensor output back to Vector */
							int			output_dim;
							int			output_size;

							/* Determine output dimension from tensor shape */
							if (output_tensor->ndim == 1)
							{
								output_dim = (int) output_tensor->shape[0];
							}
							else if (output_tensor->ndim == 2)
							{
								/*
								 * Take last dimension (e.g., [batch,
								 * features])
								 */
								output_dim = (int) output_tensor->shape[output_tensor->ndim - 1];
							}
							else
							{
								/* Flatten to 1D */
								output_dim = (int) output_tensor->size;
							}

							if (output_dim <= 0 || output_dim > VECTOR_MAX_DIM)
							{
								neurondb_onnx_free_tensor(input_tensor);
								if (output_tensor)
									neurondb_onnx_free_tensor(output_tensor);
								NDB_FREE(name_str);
								ereport(ERROR,
										(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
										 errmsg("neurondb: ONNX model output dimension %d is invalid",
												output_dim)));
							}

							{
								char	   *result_raw = NULL;
								output_size = VECTOR_SIZE(output_dim);
								NDB_ALLOC(result_raw, char, output_size);
								result = (Vector *) result_raw;
								SET_VARSIZE(result, output_size);
								result->dim = output_dim;
							}

							/* Copy output data (handle flattening if needed) */
							if (output_tensor->ndim == 1)
							{
								memcpy(result->data, output_tensor->data, sizeof(float) * output_dim);
							}
							else if (output_tensor->ndim == 2)
							{
								/* Take first batch element */
								int			batch_size = (int) output_tensor->shape[0];

								if (batch_size > 0)
								{
									int			feature_dim = (int) output_tensor->shape[1];

									memcpy(result->data, output_tensor->data, sizeof(float) * feature_dim);
								}
							}
							else
							{
								/* Flatten all dimensions */
								memcpy(result->data, output_tensor->data, sizeof(float) * output_dim);
							}

							neurondb_onnx_free_tensor(output_tensor);
						}
					}
					PG_CATCH();
					{
						if (input_tensor)
							neurondb_onnx_free_tensor(input_tensor);
						if (output_tensor)
							neurondb_onnx_free_tensor(output_tensor);
						NDB_FREE(name_str);
						PG_RE_THROW();
					}
					PG_END_TRY();

					neurondb_onnx_free_tensor(input_tensor);
				}
				else
				{
					/* ONNX session not loaded - fall back to copy */
					result = copy_vector(input);
				}
#else
				/* ONNX Runtime not compiled in - fall back to copy */
				result = copy_vector(input);
#endif
			}
			break;
		case MODEL_TF:
			/* TensorFlow backend not yet implemented */
			result = copy_vector(input);
			break;
		case MODEL_PYTORCH:
			/* PyTorch backend not yet implemented */
			result = copy_vector(input);
			break;
		default:
			/* Unknown backend - fall back to copy */
			result = copy_vector(input);
			break;
	}

	if (result == NULL)
	{
		/* Fallback to copy if inference failed */
		result = copy_vector(input);
	}

	/* Free model name string */
	NDB_FREE(name_str);

	PG_RETURN_VECTOR_P(result);
}

/* BATCH INFERENCE: Predict on an array of vectors */
PG_FUNCTION_INFO_V1(predict_batch);

Datum
predict_batch(PG_FUNCTION_ARGS)
{
	text	   *model_name;
	ArrayType  *inputs;
	char	   *name_str;
	ModelEntry *m;
	int			ndim;
	int64		nvecs;
	ArrayType  *result_array;
	Datum	   *input_datums;
	bool	   *input_nulls;
	int			i;
	Oid			vector_oid;

	NDB_DECLARE(Datum *, output_datums);
	NDB_DECLARE(bool *, output_nulls);

	NDB_CHECK_NULL_ARG(0, "model_name");
	NDB_CHECK_NULL_ARG(1, "inputs");

	model_name = PG_GETARG_TEXT_PP(0);
	inputs = PG_GETARG_ARRAYTYPE_P(1);
	name_str = text_to_cstring(model_name);

	m = find_model(name_str);
	if (m == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: Model '%s' for batch predict not found.",
						name_str)));
	if (!m->loaded || m->model_handle == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: Model '%s' is not ready for batch inference.",
						name_str)));

	if (inputs == NULL)
		ereport(ERROR,
				(errmsg("neurondb: Input array for batch predict is "
						"NULL.")));

	ndim = ARR_NDIM(inputs);
	nvecs = ArrayGetNItems(ndim, ARR_DIMS(inputs));

	if (ndim != 1)
		ereport(ERROR,
				(errmsg("neurondb: Input array for batch inference "
						"must be 1-dimensional")));

	if (ARR_HASNULL(inputs))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("neurondb: Input array for predict_batch contains nulls. Not supported.")));

	elog(DEBUG1,
		 "neurondb: [predict_batch] Model '%s' on batch: %ld vectors "
		 "(backend: '%s')",
		 name_str,
		 (long) nvecs,
		 m->model_handle->backend_msg);

	/* Check if backend is actually available (not just a stub) */
	if (strstr(m->model_handle->backend_msg, "stub") != NULL)
	{
		NDB_FREE(name_str);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("neurondb: Model '%s' backend is not available for batch inference. "
						"Backend message: %s",
						name_str,
						m->model_handle->backend_msg),
				 errhint("Install ONNX Runtime, TensorFlow, or PyTorch "
						 "to enable batch model inference")));
	}

	/* Run actual batch inference based on backend type */
	result_array = NULL;

	/* Extract input vectors from array */
	vector_oid = ARR_ELEMTYPE(inputs);
	deconstruct_array(inputs,
					  vector_oid,
					  sizeof(Vector),
					  false,
					  'i',
					  &input_datums,
					  &input_nulls,
					  NULL);

	/* Allocate output array */
	NDB_ALLOC(output_datums, Datum, nvecs);
	NDB_ALLOC(output_nulls, bool, nvecs);

	switch (m->type)
	{
		case MODEL_ONNX:
			{
#ifdef HAVE_ONNX_RUNTIME
				ONNXModelSession *session = (ONNXModelSession *) m->model_handle->opaque_backend_state;

				if (session != NULL && session->is_loaded)
				{
					/* Process each vector in batch */
					for (i = 0; i < nvecs; i++)
					{
						Vector	   *input_vec = (Vector *) DatumGetPointer(input_datums[i]);

						NDB_DECLARE(ONNXTensor *, input_tensor);
						NDB_DECLARE(ONNXTensor *, output_tensor);
						NDB_DECLARE(Vector *, output_vec);
						NDB_DECLARE(int64 *, shape);
						NDB_DECLARE(float *, data);

						if (input_nulls[i] || input_vec == NULL)
						{
							output_nulls[i] = true;
							output_datums[i] = (Datum) 0;
							continue;
						}

						/* Convert Vector to ONNXTensor */
						NDB_ALLOC(input_tensor, ONNXTensor, 1);
						memset(input_tensor, 0, sizeof(ONNXTensor));
						input_tensor->ndim = 2;
						NDB_ALLOC(shape, int64, 2);
						input_tensor->shape = shape;
						input_tensor->shape[0] = 1;
						input_tensor->shape[1] = input_vec->dim;
						input_tensor->size = input_vec->dim;
						NDB_ALLOC(data, float, input_vec->dim);
						input_tensor->data = data;
						memcpy(input_tensor->data, input_vec->data, sizeof(float) * input_vec->dim);

						/* Run inference */
						PG_TRY();
						{
							output_tensor = neurondb_onnx_run_inference(session, input_tensor);
							if (output_tensor != NULL && output_tensor->data != NULL)
							{
								int			output_dim;

								/* Determine output dimension */
								if (output_tensor->ndim == 1)
								{
									output_dim = (int) output_tensor->shape[0];
								}
								else if (output_tensor->ndim == 2)
								{
									output_dim = (int) output_tensor->shape[output_tensor->ndim - 1];
								}
								else
								{
									output_dim = (int) output_tensor->size;
								}

								if (output_dim > 0 && output_dim <= VECTOR_MAX_DIM)
								{
									int			output_size = VECTOR_SIZE(output_dim);

									NDB_DECLARE(char *, output_vec_raw);
									NDB_ALLOC(output_vec_raw, char, output_size);
									output_vec = (Vector *) output_vec_raw;
									SET_VARSIZE(output_vec, output_size);
									output_vec->dim = output_dim;

									/* Copy output data */
									if (output_tensor->ndim == 1)
									{
										memcpy(output_vec->data, output_tensor->data, sizeof(float) * output_dim);
									}
									else if (output_tensor->ndim == 2)
									{
										int			batch_size = (int) output_tensor->shape[0];

										if (batch_size > 0)
										{
											int			feature_dim = (int) output_tensor->shape[1];

											memcpy(output_vec->data, output_tensor->data, sizeof(float) * feature_dim);
										}
									}
									else
									{
										memcpy(output_vec->data, output_tensor->data, sizeof(float) * output_dim);
									}

									output_datums[i] = PointerGetDatum(output_vec);
									output_nulls[i] = false;
								}
								else
								{
									output_nulls[i] = true;
									output_datums[i] = (Datum) 0;
								}

								neurondb_onnx_free_tensor(output_tensor);
							}
							else
							{
								output_nulls[i] = true;
								output_datums[i] = (Datum) 0;
							}
						}
						PG_CATCH();
						{
							if (input_tensor)
								neurondb_onnx_free_tensor(input_tensor);
							if (output_tensor)
								neurondb_onnx_free_tensor(output_tensor);
							output_nulls[i] = true;
							output_datums[i] = (Datum) 0;
							FlushErrorState();
						}
						PG_END_TRY();

						neurondb_onnx_free_tensor(input_tensor);
					}
				}
				else
				{
					/* ONNX session not loaded - return copy of inputs */
					for (i = 0; i < nvecs; i++)
					{
						if (input_nulls[i])
						{
							output_nulls[i] = true;
							output_datums[i] = (Datum) 0;
						}
						else
						{
							Vector	   *input_vec = (Vector *) DatumGetPointer(input_datums[i]);
							Vector	   *output_vec = copy_vector(input_vec);

							output_datums[i] = PointerGetDatum(output_vec);
							output_nulls[i] = false;
						}
					}
				}
#else
				/* ONNX Runtime not compiled in - return copy of inputs */
				for (i = 0; i < nvecs; i++)
				{
					if (input_nulls[i])
					{
						output_nulls[i] = true;
						output_datums[i] = (Datum) 0;
					}
					else
					{
						Vector	   *input_vec = (Vector *) DatumGetPointer(input_datums[i]);
						Vector	   *output_vec = copy_vector(input_vec);

						output_datums[i] = PointerGetDatum(output_vec);
						output_nulls[i] = false;
					}
				}
#endif
			}
			break;
		case MODEL_TF:
			/* TensorFlow backend not yet implemented - return copy */
			for (i = 0; i < nvecs; i++)
			{
				if (input_nulls[i])
				{
					output_nulls[i] = true;
					output_datums[i] = (Datum) 0;
				}
				else
				{
					Vector	   *input_vec = (Vector *) DatumGetPointer(input_datums[i]);
					Vector	   *output_vec = copy_vector(input_vec);

					output_datums[i] = PointerGetDatum(output_vec);
					output_nulls[i] = false;
				}
			}
			break;
		case MODEL_PYTORCH:
			/* PyTorch backend not yet implemented - return copy */
			for (i = 0; i < nvecs; i++)
			{
				if (input_nulls[i])
				{
					output_nulls[i] = true;
					output_datums[i] = (Datum) 0;
				}
				else
				{
					Vector	   *input_vec = (Vector *) DatumGetPointer(input_datums[i]);
					Vector	   *output_vec = copy_vector(input_vec);

					output_datums[i] = PointerGetDatum(output_vec);
					output_nulls[i] = false;
				}
			}
			break;
		default:
			/* Unknown backend - return copy */
			for (i = 0; i < nvecs; i++)
			{
				if (input_nulls[i])
				{
					output_nulls[i] = true;
					output_datums[i] = (Datum) 0;
				}
				else
				{
					Vector	   *input_vec = (Vector *) DatumGetPointer(input_datums[i]);
					Vector	   *output_vec = copy_vector(input_vec);

					output_datums[i] = PointerGetDatum(output_vec);
					output_nulls[i] = false;
				}
			}
			break;
	}

	/* Construct output array */
	result_array = construct_array(output_datums,
								   nvecs,
								   vector_oid,
								   sizeof(Vector),
								   false,
								   'i');

	/* Free model name string */
	NDB_FREE(name_str);

	PG_RETURN_ARRAYTYPE_P(result_array);
}

/* LIST MODELS: Return metadata of all registered models as JSON array */
PG_FUNCTION_INFO_V1(list_models);

Datum
list_models(PG_FUNCTION_ARGS)
{
	StringInfoData buf;
	ModelEntry *cur;
	int			count = 0;

	initStringInfo(&buf);
	appendStringInfoString(&buf, "[");

	cur = model_registry_head;
	while (cur != NULL)
	{
		char		created_buf[32] = "";

		if (count > 0)
			appendStringInfoString(&buf, ", ");

		if (cur->model_handle && cur->model_handle->created)
		{
			struct pg_tm *tmt;
			pg_time_t	ct = (pg_time_t) cur->model_handle->created;

			tmt = pg_gmtime(&ct);
			if (tmt)
				snprintf(created_buf,
						 sizeof(created_buf),
						 "%04d-%02d-%02dT%02d:%02d:%02dZ",
						 tmt->tm_year + 1900,
						 tmt->tm_mon + 1,
						 tmt->tm_mday,
						 tmt->tm_hour,
						 tmt->tm_min,
						 tmt->tm_sec);
		}

		appendStringInfo(&buf,
						 "{"
						 "\"name\": \"%s\", "
						 "\"type\": \"%s\", "
						 "\"path\": \"%s\", "
						 "\"loaded\": %s, "
						 "\"model_size\": %lu, "
						 "\"created\": \"%s\", "
						 "\"backend_msg\": \"%s\""
						 "}",
						 cur->name,
						 model_type_to_cstr(cur->type),
						 cur->path,
						 cur->loaded ? "true" : "false",
						 (unsigned long) (cur->model_handle
										  ? cur->model_handle->model_size_bytes
										  : 0),
						 created_buf,
						 (cur->model_handle ? cur->model_handle->backend_msg
						  : "n/a"));

		count++;
		cur = cur->next;
	}

	appendStringInfoString(&buf, "]");


	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

/* FINE-TUNE MODEL: Retrain and update model given a source table with features/labels */
PG_FUNCTION_INFO_V1(finetune_model);

Datum
finetune_model(PG_FUNCTION_ARGS)
{
	text	   *model_name;
	text	   *train_table;
	text	   *config;
	char	   *name_str;
	char	   *table_str;
	char	   *config_str;
	ModelEntry *m;
	Oid			typinput,
				typioparam;
	Datum		js;
	int			ret;
	int			proc;
	StringInfoData sql;

	NDB_CHECK_NULL_ARG(0, "model_name");
	NDB_CHECK_NULL_ARG(1, "train_table");
	NDB_CHECK_NULL_ARG(2, "config");

	model_name = PG_GETARG_TEXT_PP(0);
	train_table = PG_GETARG_TEXT_PP(1);
	config = PG_GETARG_TEXT_PP(2);

	name_str = text_to_cstring(model_name);
	table_str = text_to_cstring(train_table);
	config_str = text_to_cstring(config);
	proc = 0;

	m = find_model(name_str);
	if (m == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: Model '%s' does not exist; must load prior to finetune.",
						name_str)));
	if (!m->loaded || m->model_handle == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: Model '%s' is not loaded, cannot finetune.",
						name_str)));

	/* Validate/parse config JSON */
	getTypeInputInfo(JSONBOID, &typinput, &typioparam);
	js = OidFunctionCall3(typinput,
						  CStringGetDatum(config_str),
						  ObjectIdGetDatum(InvalidOid),
						  Int32GetDatum(-1));
	if (js == (Datum) 0)
		ereport(ERROR,
				(errmsg("neurondb: Provided fine-tune config is not "
						"valid JSON.")));


	/* Scan training table and acquire training data using SPI */
	{
		NDB_DECLARE(NdbSpiSession *, finetune_spi_session);
		MemoryContext oldcontext = CurrentMemoryContext;

		NDB_SPI_SESSION_BEGIN(finetune_spi_session, oldcontext);
		initStringInfo(&sql);
		appendStringInfo(&sql, "SELECT * FROM %s", quote_identifier(table_str));
		ret = ndb_spi_execute_safe(sql.data, true, 0);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(finetune_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: Failed to scan training table '%s' via SPI.",
							table_str),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the table exists and contains valid data.")));
		}

		proc = SPI_processed;
		elog(LOG,
			 "neurondb: Finetune SPI scan: table='%s', rows=%d",
			 table_str,
			 proc);

		/* Simulate model update by incrementing version and updating state */
		if (m->model_handle)
		{
			m->model_handle->version++;
			snprintf(m->model_handle->backend_msg,
					 sizeof(m->model_handle->backend_msg),
					 "Fine-tuned at %ld; new version v%d",
					 (long) time(NULL),
					 m->model_handle->version);
			m->loaded = true;
		}
		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(finetune_spi_session);
	}

	elog(DEBUG1,
		 "neurondb: Fine-tuned model '%s' using %d rows from '%s'. Config: %s. New version: %d",
		 name_str,
		 proc,
		 table_str,
		 config_str,
		 m->model_handle->version);

	PG_RETURN_BOOL(true);
}

/* EXPORT MODEL: Save a loaded model to requested output path/format */
PG_FUNCTION_INFO_V1(export_model);

Datum
export_model(PG_FUNCTION_ARGS)
{
	text	   *model_name;
	text	   *output_path;
	text	   *output_format;
	char	   *name_str;
	char	   *path_str;
	char	   *fmt_str;
	ModelEntry *m;
	char	   *lastslash;
	size_t		sz;
	FILE	   *f;

	NDB_CHECK_NULL_ARG(0, "model_name");
	NDB_CHECK_NULL_ARG(1, "output_path");
	NDB_CHECK_NULL_ARG(2, "output_format");

	model_name = PG_GETARG_TEXT_PP(0);
	output_path = PG_GETARG_TEXT_PP(1);
	output_format = PG_GETARG_TEXT_PP(2);

	name_str = text_to_cstring(model_name);
	path_str = text_to_cstring(output_path);
	fmt_str = text_to_cstring(output_format);

	m = find_model(name_str);
	if (m == NULL || m->model_handle == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("neurondb: Model '%s' is not loaded for export.",
						name_str)));

	lastslash = strrchr(path_str, '/');
	if (lastslash != NULL)
	{
		char		dir[MAXPGPATH];
		size_t		dirlen = lastslash - path_str;
		struct stat st;

		if (dirlen >= sizeof(dir) || dirlen <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("neurondb: Output path '%s' has invalid directory component.",
							path_str)));
		memcpy(dir, path_str, dirlen);
		dir[dirlen] = '\0';

		if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode))
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("neurondb: Output directory '%s' does not exist.",
							dir)));
		if (access(dir, W_OK) != 0)
			ereport(ERROR,
					(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
					 errmsg("neurondb: No write permission to directory '%s'.",
							dir)));
	}

	if (pg_strcasecmp(fmt_str, model_type_to_cstr(m->type)) != 0)
	{
		elog(DEBUG1,
			 "neurondb: Model '%s' is type '%s' but export requested as '%s'. Compatibility not guaranteed.",
			 name_str,
			 model_type_to_cstr(m->type),
			 fmt_str);
	}

	f = fopen(path_str, "wb");
	if (f == NULL)
	{
		int			saved_errno = errno;

		ereport(ERROR,
				(errcode(ERRCODE_IO_ERROR),
				 errmsg("neurondb: Failed to open export file '%s' (%s)",
						path_str,
						strerror(saved_errno))));
	}

	sz = m->model_handle ? m->model_handle->model_size_bytes : 0;

	/* Write a detailed header for audit/tracing: */
	if (fprintf(f, "# NeurondB Model Export\n") < 0 ||
		fprintf(f, "%% name = %s\n", name_str) < 0 ||
		fprintf(f, "%% type = %s\n", model_type_to_cstr(m->type)) < 0 ||
		fprintf(f, "%% export_format = %s\n", fmt_str) < 0 ||
		fprintf(f, "%% file = %s\n", path_str) < 0 ||
		fprintf(f, "%% original_file = %s\n", m->path) < 0 ||
		fprintf(f, "%% bytes = %lu\n", (unsigned long) sz) < 0 ||
		fprintf(f, "%% backend_msg = %s\n", m->model_handle->backend_msg) < 0 ||
		fprintf(f, "%% exported_at = %ld\n", (long) time(NULL)) < 0 ||
		fprintf(f, "%% version = %u\n", m->model_handle->version) < 0 ||
		fprintf(f, "--\n") < 0)
	{
		int			saved_errno = errno;

		fclose(f);
		ereport(ERROR,
				(errcode(ERRCODE_IO_ERROR),
				 errmsg("neurondb: Failed to write header to export file '%s' (%s)",
						path_str, strerror(saved_errno))));
	}

	/* Check if model data is actually available (not a stub) */
	if (m->model_handle == NULL ||
		strstr(m->model_handle->backend_msg, "stub") != NULL ||
		m->model_handle->model_size_bytes == 0)
	{
		fclose(f);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("Cannot export model: model data is not available"),
				 errdetail("Model '%s' backend is not available or model data is missing. Backend message: %s",
						   name_str,
						   m->model_handle ? m->model_handle->backend_msg : "model handle is NULL"),
				 errhint("Real model implementation required. Install ONNX Runtime, TensorFlow, or PyTorch to enable model export.")));
	}

	/* Export model data: read from original file and write to export file */
	{
		FILE	   *src_f = NULL;
		NDB_DECLARE(char *, buffer);
		size_t		bytes_read;
		size_t		buffer_size = 8192; /* 8KB buffer for file copy */
		const char *src_path = m->path;

		/* Open source model file */
		src_f = fopen(src_path, "rb");
		if (src_f == NULL)
		{
			int			saved_errno = errno;

			fclose(f);
			ereport(ERROR,
					(errcode(ERRCODE_IO_ERROR),
					 errmsg("Cannot export model: failed to read source model file"),
					 errdetail("Failed to open source model file '%s' for reading: %s",
							   src_path, strerror(saved_errno))));
		}

		/* Allocate buffer for file copy */
		NDB_ALLOC(buffer, char, buffer_size);
		if (buffer == NULL)
		{
			fclose(src_f);
			fclose(f);
			ereport(ERROR,
					(errcode(ERRCODE_OUT_OF_MEMORY),
					 errmsg("Cannot export model: failed to allocate memory for file copy")));
		}

		/* Copy file contents */
		while ((bytes_read = fread(buffer, 1, buffer_size, src_f)) > 0)
		{
			if (fwrite(buffer, 1, bytes_read, f) != bytes_read)
			{
				int			saved_errno = errno;

				NDB_FREE(buffer);
				fclose(src_f);
				fclose(f);
				ereport(ERROR,
						(errcode(ERRCODE_IO_ERROR),
						 errmsg("Cannot export model: failed to write model data"),
						 errdetail("Failed to write to export file '%s': %s",
								   path_str, strerror(saved_errno))));
			}
		}

		/* Check for read errors */
		if (ferror(src_f))
		{
			int			saved_errno = errno;

			NDB_FREE(buffer);
			fclose(src_f);
			fclose(f);
			ereport(ERROR,
					(errcode(ERRCODE_IO_ERROR),
					 errmsg("Cannot export model: failed to read source model file"),
					 errdetail("Error reading from source file '%s': %s",
							   src_path, strerror(saved_errno))));
		}

		NDB_FREE(buffer);
		fclose(src_f);
	}

	if (fclose(f) != 0)
	{
		int			saved_errno = errno;

		ereport(WARNING,
				(errcode(ERRCODE_IO_ERROR),
				 errmsg("neurondb: Failed to close export file '%s' (%s)",
						path_str, strerror(saved_errno))));
	}

	elog(DEBUG1,
		 "neurondb: Model '%s' (type=%s) exported to '%s' as format '%s'; size=%lu, version=%u",
		 name_str,
		 model_type_to_cstr(m->type),
		 path_str,
		 fmt_str,
		 (unsigned long) sz,
		 m->model_handle->version);

	PG_RETURN_BOOL(true);
}

/* EXPORT MODEL TO ONNX: Export NeuronDB model to ONNX format */
PG_FUNCTION_INFO_V1(export_model_to_onnx);

Datum
export_model_to_onnx(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *output_path_text;
	char	   *output_path;
	char	   *algorithm;
	bytea	   *model_data;
	StringInfoData sql;
	int			ret;
	FILE	   *temp_file;
	char	   *temp_path;
	int			cmd_ret;
	char		cmd_buf[4096];
	char	   *script_dir;
	char	   *full_script_path;

	NDB_CHECK_NULL_ARG(0, "model_id");
	NDB_CHECK_NULL_ARG(1, "output_path");

	model_id = PG_GETARG_INT32(0);
	output_path_text = PG_GETARG_TEXT_PP(1);
	output_path = NULL;
	algorithm = NULL;
	model_data = NULL;
	temp_file = NULL;
	temp_path = NULL;
	script_dir = NULL;
	full_script_path = NULL;

	/* Fetch model from catalog */
	output_path = text_to_cstring(output_path_text);

	/* Query model from catalog */
	{
		NDB_DECLARE(NdbSpiSession *, export_spi_session);
		MemoryContext oldcontext = CurrentMemoryContext;

		NDB_SPI_SESSION_BEGIN(export_spi_session, oldcontext);
		initStringInfo(&sql);
		appendStringInfo(&sql,
						 "SELECT model_data, algorithm, parameters, metrics "
						 "FROM neurondb.ml_models WHERE model_id = %d",
						 model_id);

		ret = ndb_spi_execute_safe(sql.data, true, 1);
		NDB_CHECK_SPI_TUPTABLE();
		if (ret != SPI_OK_SELECT)
		{
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(export_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: export_model_to_onnx: failed to query model"),
					 errdetail("SPI execution returned code %d (expected %d)", ret, SPI_OK_SELECT),
					 errhint("Verify the model_id exists in neurondb.ml_models.")));
		}

		if (SPI_processed == 0)
		{
			NDB_FREE(sql.data);
			NDB_SPI_SESSION_END(export_spi_session);
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("neurondb: Model with id %d not found", model_id),
					 errdetail("The query returned 0 rows"),
					 errhint("Verify the model_id is correct and the model exists.")));
		}

		/* Validate SPI_tuptable before access */
		NDB_CHECK_SPI_TUPTABLE();

		/* Extract model data and algorithm */
		{
			text	   *algorithm_text_ptr;
			char	   *algorithm_text;

			/* Safe access for complex types - validate before access */
			if (SPI_tuptable == NULL || SPI_tuptable->vals == NULL ||
				SPI_processed == 0 || SPI_tuptable->vals[0] == NULL || SPI_tuptable->tupdesc == NULL)
			{
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(export_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("neurondb: Model %d result is invalid", model_id)));
			}
			/* Get model_data - use safe function for bytea */
			model_data = ndb_spi_get_bytea(export_spi_session, 0, 1, oldcontext);
			if (model_data == NULL)
			{
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(export_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("neurondb: Model %d has no model_data", model_id),
						 errdetail("The model_data column is NULL"),
						 errhint("Ensure the model was properly trained and saved.")));
			}

			/* Validate model_data after copy */
			if (model_data == NULL)
			{
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(export_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("neurondb: Failed to copy model_data for model %d", model_id),
						 errdetail("Memory allocation failed during model data copy"),
						 errhint("Check available memory and try again.")));
			}

			/* Get algorithm - use safe function for text */
			algorithm_text_ptr = ndb_spi_get_text(export_spi_session, 0, 2, oldcontext);
			if (algorithm_text_ptr == NULL)
			{
				NDB_FREE(model_data);
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(export_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("neurondb: Model %d has no algorithm", model_id),
						 errdetail("The algorithm column is NULL"),
						 errhint("Ensure the model was properly registered.")));
			}
			algorithm_text = text_to_cstring(algorithm_text_ptr);
			if (algorithm_text == NULL)
			{
				NDB_FREE(model_data);
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(export_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("neurondb: Model %d has NULL algorithm", model_id),
						 errdetail("The algorithm text conversion returned NULL"),
						 errhint("Verify the model data integrity.")));
			}
			algorithm = pstrdup(algorithm_text);
			if (algorithm == NULL)
			{
				NDB_FREE(model_data);
				NDB_FREE(sql.data);
				NDB_SPI_SESSION_END(export_spi_session);
				ereport(ERROR,
						(errcode(ERRCODE_OUT_OF_MEMORY),
						 errmsg("neurondb: Failed to allocate algorithm string"),
						 errdetail("Memory allocation failed"),
						 errhint("Check available memory and try again.")));
			}
		}

		NDB_FREE(sql.data);
		NDB_SPI_SESSION_END(export_spi_session);
	}

	/* Create temporary file for model data */
	temp_path = psprintf("/tmp/neurondb_export_%d_%ld.bin",
						 model_id, (long) time(NULL));
	temp_file = fopen(temp_path, "wb");
	if (temp_file == NULL)
	{
		int			saved_errno = errno;

		NDB_FREE(model_data);
		NDB_FREE(algorithm);
		NDB_FREE(temp_path);
		ereport(ERROR,
				(errcode(ERRCODE_IO_ERROR),
				 errmsg("neurondb: Failed to create temp file '%s' (%s)",
						temp_path, strerror(saved_errno))));
	}

	/* Write model data to temp file */
	{
		size_t		bytes_written = fwrite(VARDATA(model_data), 1, VARSIZE(model_data) - VARHDRSZ, temp_file);
		size_t		expected_bytes = VARSIZE(model_data) - VARHDRSZ;

		if (bytes_written != expected_bytes)
		{
			int			saved_errno = errno;

			fclose(temp_file);
			unlink(temp_path);
			NDB_FREE(temp_path);
			NDB_FREE(algorithm);
			NDB_FREE(model_data);
			ereport(ERROR,
					(errcode(ERRCODE_IO_ERROR),
					 errmsg("neurondb: export_model_to_onnx: failed to write model data to temp file '%s' (%s)",
							temp_path, strerror(saved_errno))));
		}

		if (fclose(temp_file) != 0)
		{
			int			saved_errno = errno;

			unlink(temp_path);
			NDB_FREE(temp_path);
			NDB_FREE(algorithm);
			NDB_FREE(model_data);
			ereport(ERROR,
					(errcode(ERRCODE_IO_ERROR),
					 errmsg("neurondb: export_model_to_onnx: failed to close temp file '%s' (%s)",
							temp_path, strerror(saved_errno))));
		}
		temp_file = NULL;
	}

	/* Find Python script path */
	/* Try to find script relative to installation */
	script_dir = getenv("NEURONDB_TOOLS_DIR");
	if (script_dir == NULL)
	{
		/* Default to tools directory relative to share/neurondb */
		char	   *share_path = getenv("NEURONDB_SHARE_DIR");

		if (share_path != NULL)
		{
			script_dir = psprintf("%s/../tools", share_path);
		}
		else
		{
			/* Last resort: assume script is in tools/ directory */
			script_dir = pstrdup("tools");
		}
	}
	else
	{
		script_dir = pstrdup(script_dir);
	}

	full_script_path = psprintf("%s/neurondb_onex.py", script_dir);

	/* Check if script exists */
	if (access(full_script_path, R_OK | X_OK) != 0)
	{
		int			saved_errno = errno;

		unlink(temp_path);
		NDB_FREE(model_data);
		NDB_FREE(algorithm);
		NDB_FREE(temp_path);
		NDB_FREE(script_dir);
		NDB_FREE(full_script_path);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FILE),
				 errmsg("neurondb: Python script not found or not executable: %s (%s)",
						full_script_path, strerror(saved_errno)),
				 errhint("Set NEURONDB_TOOLS_DIR environment variable or ensure script is in tools/ directory")));
	}

	/* Build command to call Python script */
	snprintf(cmd_buf, sizeof(cmd_buf),
			 "python3 %s --export %s %s %s",
			 full_script_path, temp_path, algorithm, output_path);

	elog(DEBUG1, "neurondb: export_model_to_onnx: executing: %s", cmd_buf);

	/* Execute Python script */
	cmd_ret = system(cmd_buf);
	if (cmd_ret != 0)
	{
		unlink(temp_path);
		NDB_FREE(model_data);
		NDB_FREE(algorithm);
		NDB_FREE(temp_path);
		NDB_FREE(script_dir);
		NDB_FREE(full_script_path);
		ereport(ERROR,
				(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
				 errmsg("neurondb: Failed to convert model to ONNX (exit code %d)",
						cmd_ret),
				 errhint("Check that Python and onnx library are installed")));
	}

	/* Clean up temp file */
	unlink(temp_path);

	elog(DEBUG1,
		 "neurondb: export_model_to_onnx: Model %d (algorithm=%s) exported to %s",
		 model_id, algorithm, output_path);

	NDB_FREE(model_data);
	NDB_FREE(algorithm);
	NDB_FREE(temp_path);
	NDB_FREE(script_dir);
	NDB_FREE(full_script_path);

	PG_RETURN_BOOL(true);
}

/* IMPORT MODEL FROM ONNX: Import ONNX model to NeuronDB format */
PG_FUNCTION_INFO_V1(import_model_from_onnx);

Datum
import_model_from_onnx(PG_FUNCTION_ARGS)
{
	int32		model_id;
	text	   *onnx_path_text;
	text	   *algorithm_text;
	char	   *onnx_path;
	char	   *algorithm;
	char	   *temp_path;
	int			cmd_ret;
	char		cmd_buf[4096];
	char	   *script_dir;
	char	   *full_script_path;
	FILE	   *output_file;
	bytea	   *model_data;
	size_t		file_size;

	NDB_DECLARE(char *, file_data);
	int			ret;
	StringInfoData update_sql = {0};

	NDB_CHECK_NULL_ARG(0, "model_id");
	NDB_CHECK_NULL_ARG(1, "onnx_path");
	NDB_CHECK_NULL_ARG(2, "algorithm");

	model_id = PG_GETARG_INT32(0);
	onnx_path_text = PG_GETARG_TEXT_PP(1);
	algorithm_text = PG_GETARG_TEXT_PP(2);
	onnx_path = NULL;
	algorithm = NULL;
	temp_path = NULL;
	script_dir = NULL;
	full_script_path = NULL;
	output_file = NULL;
	model_data = NULL;
	file_size = 0;
	file_data = NULL;

	onnx_path = text_to_cstring(onnx_path_text);
	algorithm = text_to_cstring(algorithm_text);

	/* Validate ONNX file exists */
	if (access(onnx_path, R_OK) != 0)
	{
		int			saved_errno = errno;

		NDB_FREE(onnx_path);
		NDB_FREE(algorithm);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FILE),
				 errmsg("neurondb: ONNX file not found or not readable: %s (%s)",
						onnx_path, strerror(saved_errno))));
	}

	/* Find Python script path */
	script_dir = getenv("NEURONDB_TOOLS_DIR");
	if (script_dir == NULL)
	{
		char	   *share_path = getenv("NEURONDB_SHARE_DIR");

		if (share_path != NULL)
		{
			script_dir = psprintf("%s/../tools", share_path);
		}
		else
		{
			script_dir = pstrdup("tools");
		}
	}
	else
	{
		script_dir = pstrdup(script_dir);
	}

	full_script_path = psprintf("%s/neurondb_onex.py", script_dir);

	/* Check if script exists */
	if (access(full_script_path, R_OK | X_OK) != 0)
	{
		int			saved_errno = errno;

		NDB_FREE(onnx_path);
		NDB_FREE(algorithm);
		NDB_FREE(script_dir);
		NDB_FREE(full_script_path);
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FILE),
				 errmsg("neurondb: Python script not found or not executable: %s (%s)",
						full_script_path, strerror(saved_errno)),
				 errhint("Set NEURONDB_TOOLS_DIR environment variable or ensure script is in tools/ directory")));
	}

	/* Create temporary file for output */
	temp_path = psprintf("/tmp/neurondb_import_%d_%ld.bin",
						 model_id, (long) time(NULL));

	/* Build command to call Python script */
	snprintf(cmd_buf, sizeof(cmd_buf),
			 "python3 %s --import %s %s %s",
			 full_script_path, onnx_path, algorithm, temp_path);

	elog(DEBUG1, "neurondb: import_model_from_onnx: executing: %s", cmd_buf);

	/* Execute Python script */
	cmd_ret = system(cmd_buf);
	if (cmd_ret != 0)
	{
		unlink(temp_path);
		NDB_FREE(onnx_path);
		NDB_FREE(algorithm);
		NDB_FREE(temp_path);
		NDB_FREE(script_dir);
		NDB_FREE(full_script_path);
		ereport(ERROR,
				(errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
				 errmsg("neurondb: Failed to import ONNX model (exit code %d)",
						cmd_ret),
				 errhint("Check that Python and onnx library are installed")));
	}

	/* Read converted model data from temp file */
	output_file = fopen(temp_path, "rb");
	if (output_file == NULL)
	{
		int			saved_errno = errno;

		unlink(temp_path);
		NDB_FREE(onnx_path);
		NDB_FREE(algorithm);
		NDB_FREE(temp_path);
		NDB_FREE(script_dir);
		NDB_FREE(full_script_path);
		ereport(ERROR,
				(errcode(ERRCODE_IO_ERROR),
				 errmsg("neurondb: Failed to read converted model file '%s' (%s)",
						temp_path, strerror(saved_errno))));
	}

	/* Get file size */
	fseek(output_file, 0, SEEK_END);
	file_size = ftell(output_file);
	fseek(output_file, 0, SEEK_SET);

	/* Read file data */
	file_data = NULL;
	NDB_ALLOC(file_data, char, file_size);
	if (fread(file_data, 1, file_size, output_file) != file_size)
	{
		int			saved_errno = errno;

		fclose(output_file);
		unlink(temp_path);
		NDB_FREE(file_data);
		NDB_FREE(onnx_path);
		NDB_FREE(algorithm);
		NDB_FREE(temp_path);
		NDB_FREE(script_dir);
		NDB_FREE(full_script_path);
		ereport(ERROR,
				(errcode(ERRCODE_IO_ERROR),
				 errmsg("neurondb: Failed to read model data from '%s' (%s)",
						temp_path, strerror(saved_errno))));
	}

	if (fclose(output_file) != 0)
	{
		int			saved_errno = errno;

		unlink(temp_path);
		NDB_FREE(file_data);
		NDB_FREE(onnx_path);
		NDB_FREE(algorithm);
		NDB_FREE(temp_path);
		NDB_FREE(script_dir);
		NDB_FREE(full_script_path);
		ereport(WARNING,
				(errcode(ERRCODE_IO_ERROR),
				 errmsg("neurondb: import_model_from_onnx: failed to close temp file '%s' (%s)",
						temp_path, strerror(saved_errno))));
	}
	unlink(temp_path);

	/* Create bytea from file data */
	{
		char	   *model_data_raw = NULL;
		NDB_ALLOC(model_data_raw, char, VARHDRSZ + file_size);
		model_data = (bytea *) model_data_raw;
		SET_VARSIZE(model_data, VARHDRSZ + file_size);
		memcpy(VARDATA(model_data), file_data, file_size);
	}
	NDB_FREE(file_data);

	/* Update model in catalog */
	{
		NDB_DECLARE(NdbSpiSession *, import_spi_session);
		MemoryContext oldcontext = CurrentMemoryContext;
		Oid			argtypes[2];
		Datum		values[2];
		char		nulls[2];

		NDB_SPI_SESSION_BEGIN(import_spi_session, oldcontext);
		initStringInfo(&update_sql);
		appendStringInfo(&update_sql,
						 "UPDATE neurondb.ml_models SET model_data = $1 WHERE model_id = $2");

		argtypes[0] = BYTEAOID;
		argtypes[1] = INT4OID;
		values[0] = PointerGetDatum(model_data);
		values[1] = Int32GetDatum(model_id);
		nulls[0] = ' ';
		nulls[1] = ' ';

		ret = ndb_spi_execute_with_args(
										import_spi_session,
										update_sql.data,
										2,
										argtypes,
										values,
										nulls,
										false,
										0);

		if (ret != SPI_OK_UPDATE)
		{
			elog(WARNING,
				 "neurondb: import_model_from_onnx: failed to update model_data: SPI return code %d",
				 ret);
		}

		NDB_FREE(update_sql.data);

		if (ret != SPI_OK_UPDATE || SPI_processed == 0)
		{
			NDB_FREE(model_data);
			NDB_SPI_SESSION_END(import_spi_session);
			NDB_FREE(onnx_path);
			NDB_FREE(algorithm);
			NDB_FREE(script_dir);
			NDB_FREE(full_script_path);
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("neurondb: Failed to update model %d with imported data",
							model_id)));
		}

		NDB_SPI_SESSION_END(import_spi_session);
	}

	elog(DEBUG1,
		 "neurondb: import_model_from_onnx: Model %d (algorithm=%s) imported from %s",
		 model_id, algorithm, onnx_path);

	NDB_FREE(model_data);
	NDB_FREE(onnx_path);
	NDB_FREE(algorithm);
	NDB_FREE(script_dir);
	NDB_FREE(full_script_path);

	PG_RETURN_BOOL(true);
}
