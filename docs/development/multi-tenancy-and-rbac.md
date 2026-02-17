# Multi-Tenancy and RBAC

## Row-level security (PostgreSQL)

- Enable RLS on tables that hold tenant-specific data (e.g. `neurondb.ml_models`, application tables in NeuronDesktop).
- Add a `tenant_id` (or `project_id`) column where appropriate; create policies that restrict `SELECT/INSERT/UPDATE/DELETE` to the current user's tenant (e.g. `WHERE tenant_id = current_setting('app.tenant_id')::uuid`).
- Set `app.tenant_id` (or equivalent) at the start of each session from the authenticated user's tenant.

## Tenant isolation

- NeuronAgent / NeuronDesktop: Resolve tenant from JWT or API key and pass it to DB (e.g. via `SET app.tenant_id` before queries). Ensure no cross-tenant data leakage in list/detail endpoints.
- NeuronMCP: If multi-tenant, scope tool results by tenant (e.g. SQL runs in a session with tenant_id set).

## Resource quotas per tenant

- Track usage per tenant (API calls, LLM tokens, storage). Enforce limits in middleware or before expensive operations; return 429 when over quota.
- Store quotas and usage in DB or Redis; update on each request or periodically.

## Fine-grained RBAC

- Define roles (e.g. admin, analyst, viewer) and permissions (e.g. train_model, run_sql, view_agents). Check permission in API handlers before performing the action.
- Store role-permission mapping and user-role assignment in the database; cache in memory for fast checks.

## Audit trail

- Log all data access and admin actions with tenant_id, user_id, resource, action, timestamp. Use for compliance and debugging.
