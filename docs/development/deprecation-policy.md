# Deprecation policy (SQL surface and C symbols)

## SQL functions and types

1. **Announce:** Add a clear `COMMENT ON FUNCTION` / release note stating the replacement and the last release where the old name remains available.
2. **Minimum horizon:** Prefer at least **one** minor release where both old and new entry points work (wrappers are acceptable).
3. **Remove:** Drop deprecated objects only in a **major** or explicitly marked breaking release, with upgrade script notes in `sql/neurondb--*--*.sql`.

## C extension internals

- External callers should rely on **SQL** and stable extension APIs, not `extern` symbols from `neurondb.so`.
- When renaming C entry points, keep SQL `CREATE FUNCTION` pointing at the new symbol and provide SQL wrappers for compatibility during the transition.

## Process

- Track deprecations in release notes and in [engineering-debt.md](engineering-debt.md) until removed.
