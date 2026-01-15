/*-------------------------------------------------------------------------
 *
 * runtime_coordinator.go
 *    Distributed coordinator integration for runtime
 *
 * Provides methods to set and manage distributed coordinator.
 *
 * Copyright (c) 2024-2026, neurondb, Inc. <support@neurondb.ai>
 *
 * IDENTIFICATION
 *    NeuronAgent/internal/agent/runtime_coordinator.go
 *
 *-------------------------------------------------------------------------
 */

package agent

/* SetCoordinator sets the distributed coordinator */
func (r *Runtime) SetCoordinator(coordinator interface{}) {
	r.coordinator = coordinator
}

/* GetCoordinator returns the distributed coordinator */
func (r *Runtime) GetCoordinator() interface{} {
	return r.coordinator
}


