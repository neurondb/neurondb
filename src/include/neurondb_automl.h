#ifndef NEURONDB_AUTOML_H
#define NEURONDB_AUTOML_H

#include "postgres.h"

/* AutoML backend indicator */
typedef enum AutoMLBackendType
{
	AUTOML_BACKEND_CPU = 0,
	AUTOML_BACKEND_GPU = 1
} AutoMLBackendType;

/* GUC variables are now in neurondb_guc.h */
#include "neurondb_guc.h"

/* GUC initialization is now centralized in neurondb_guc.c */
extern AutoMLBackendType neurondb_automl_choose_backend(const char *algorithm);

#endif							/* NEURONDB_AUTOML_H */
