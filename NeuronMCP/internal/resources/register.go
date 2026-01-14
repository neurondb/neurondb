/*-------------------------------------------------------------------------
 *
 * register.go
 *    Resource registration
 *
 * Registers all built-in resources with the resource manager.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronMCP/internal/resources/register.go
 *
 *-------------------------------------------------------------------------
 */

package resources

import "github.com/neurondb/NeuronMCP/internal/database"

/* RegisterAllResources registers all built-in resources */
func RegisterAllResources(manager *Manager, db *database.Database) {
	/* Register datasets resource */
	manager.Register(NewDatasetsResource(db))

	/* Register collections resource */
	manager.Register(NewCollectionsResource(db))
}


