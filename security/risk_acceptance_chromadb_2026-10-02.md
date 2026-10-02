# Temporary Risk Acceptance: ChromaDB Advisories

**Status:** Accepted temporarily with deployment restrictions  
**Recorded:** 2026-10-02  
**Risk owner:** AEM project security owner  
**Component:** `chromadb` 1.5.9, selected by `chromadb>=0.5.0`

## Findings

CI reports the following unique ChromaDB advisory IDs (the output lists
`PYSEC-2026-311` twice):

- `PYSEC-2026-311`
- `PYSEC-2026-3813`
- `PYSEC-2026-3814`
- `PYSEC-2026-3815`

The advisory data reviewed for this acceptance lists no fixed versions. This
decision does not treat another ChromaDB version as a fix.

## Scope and Rationale

The current ingestion code uses `chromadb.PersistentClient` in
`src/asa/ingestion/embed.py`. The current `docker-compose.yml` starts only the
backend and frontend; it does not start a ChromaDB server or publish a ChromaDB
port. A repository search found no `chroma run`, `HttpClient`, or server
configuration.

The reported vulnerable HTTP server/API paths are not part of the intended
embedded-only runtime. The package itself remains vulnerable, so this is a
temporary, topology-specific acceptance, not a vulnerability fix. This
acceptance is invalid if a ChromaDB HTTP server/API is enabled or exposed.

## Required Controls

- Use `PersistentClient` only. Do not run `chroma run`, use `HttpClient`, or add
  a ChromaDB server service.
- Do not expose ChromaDB server ports or enable server mode in deployment.
- Keep the CI audit ignores limited to the four IDs above. All other findings
  must continue to fail the dependency-scan job.
- Reassess before production deployment and whenever the ChromaDB access mode
  or deployment topology changes.
- Remove the CI ignores and rerun the full dependency audit once a fixed
  upstream release is available.

## Review

The risk owner must review this acceptance before production deployment, after
any relevant architecture change, and when upstream publishes a fix. If the
HTTP server/API is required, replace or upgrade ChromaDB before enabling that
topology.