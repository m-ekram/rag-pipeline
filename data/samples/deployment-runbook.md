# Deployment Runbook

## Environments

| Environment | Cluster        | Branch  | Approval        |
|-------------|----------------|---------|-----------------|
| dev         | orbit-dev      | any     | none            |
| staging     | orbit-stg      | main    | automatic       |
| production  | orbit-prod     | tagged  | two reviewers   |

Production deploys are gated on a signed git tag matching `v*.*.*`. Unsigned
tags are rejected by the release pipeline with `RELEASE_TAG_UNSIGNED`.

## Required environment variables

Every service must have these set or it will refuse to start:

- `ORBIT_ENV` - one of `dev`, `staging`, `production`
- `DATABASE_URL` - full Postgres DSN, must include `sslmode=require` in prod
- `REDIS_URL` - used for rate limiting and session cache
- `LOG_LEVEL` - defaults to `info`; use `debug` only temporarily, it roughly
  triples log volume and the log budget is not generous
- `OTEL_EXPORTER_OTLP_ENDPOINT` - traces are dropped silently if unset

## Deploy procedure

1. Confirm CI is green on the commit you intend to ship.
2. Tag the release: `git tag -s v1.4.0 -m "release 1.4.0"` and push the tag.
3. The pipeline builds the image and pushes to the registry as
   `registry.internal/orbit/<service>:v1.4.0`.
4. Run database migrations **before** shifting traffic:
   `orbitctl migrate --env production --dry-run` first, then without the flag.
5. Deploy to a 10% canary: `orbitctl deploy --canary 10 --tag v1.4.0`.
6. Watch the canary for at least 15 minutes. Promote with
   `orbitctl promote --tag v1.4.0` or roll back.

## Rollback

Rollback is a redeploy of the previous tag, not a git revert:

    orbitctl deploy --tag v1.3.9 --immediate

The `--immediate` flag skips the canary stage and shifts 100% of traffic at
once. Use it only during an active incident.

Migrations are **not** rolled back automatically. Every migration must be
backward compatible with the previous release - this is why the migration
policy requires expand-then-contract: add the new column, deploy code that
writes both, backfill, then drop the old column in a later release.

## Rollback triggers

Roll back without further discussion if any of these hold for 5 minutes after a
deploy:

- Error rate above 2% on any route
- p99 latency above 800 ms on the checkout path
- Any increase in `ERR_UPSTREAM_TIMEOUT` beyond baseline
- Queue depth growing monotonically for 10 minutes

## Health checks

The platform probes `/healthz` every 10 seconds. Three consecutive failures
remove the pod from the load balancer; ten consecutive failures restart it.
`/healthz` must not check downstream dependencies - a slow database should
not take the whole fleet out of rotation. Use `/readyz` for dependency checks.
