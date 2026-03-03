# API Versioning and Error Standardization

## Versioning

- NeuronAgent and NeuronDesktop expose APIs under `/api/v1/`. New breaking changes should go to `/api/v2/` with a deprecation period for v1.
- Document version in OpenAPI spec and response headers (e.g. `X-API-Version: 1`).
- Provide a deprecation policy: e.g. v1 supported for 12 months after v2 release, then sunset.

## Error format

- Use a consistent JSON shape for errors:
  ```json
  {
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "Human-readable message",
      "details": {},
      "request_id": "uuid"
    }
  }
  ```
- Map HTTP status (400, 401, 403, 404, 429, 500) to appropriate codes. Use stable codes (e.g. `RATE_LIMIT_EXCEEDED`) for client handling.

## Pagination

- List endpoints should support `limit` and `offset` (or `cursor`). Return metadata: `total`, `limit`, `offset`, `has_more`.
- Default limit (e.g. 20), max limit (e.g. 100) to avoid overload.

## OpenAPI

- NeuronAgent has OpenAPI spec; NeuronDesktop and NeuronMCP should expose machine-readable specs (Swagger/OpenAPI 3) for all public endpoints.
- Generate client SDKs from OpenAPI (e.g. for Python, TypeScript, Go).
