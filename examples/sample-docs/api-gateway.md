# Orbit API Gateway - Reference

## Authentication

Every request must carry a bearer token in the `Authorization` header:

    Authorization: Bearer <token>

Tokens are issued by the `/v1/auth/token` endpoint and expire after **3600
seconds**. Refresh tokens live for 30 days and are single-use: presenting a
refresh token invalidates it and returns a new pair. Clients that reuse a spent
refresh token receive `ERR_TOKEN_REPLAY` and the entire token family is revoked,
which forces a full re-login.

Service-to-service calls should use mTLS instead of bearer tokens. Provision a
client certificate through the `orbitctl cert issue` command; certificates are
valid for 90 days and rotate automatically at 75 days.

## Rate limiting

Rate limits are applied per API key using a sliding window counter.

| Tier       | Requests / minute | Burst | Concurrent connections |
|------------|-------------------|-------|------------------------|
| Free       | 60                | 10    | 5                      |
| Standard   | 600               | 100   | 50                     |
| Enterprise | 6000              | 1000  | 500                    |

When a client exceeds its limit the gateway returns HTTP 429 with a
`Retry-After` header in seconds. The response body carries the error code
`ERR_RATE_LIMITED` and the window reset timestamp.

Rate limit state is held in Redis. If Redis is unreachable the gateway fails
open for Enterprise keys and fails closed for everything else - a deliberate
tradeoff that keeps paying traffic flowing during a cache outage.

## Error codes

| Code                | HTTP | Meaning                                       |
|---------------------|------|-----------------------------------------------|
| ERR_BAD_REQUEST     | 400  | Malformed payload or missing required field   |
| ERR_UNAUTHENTICATED | 401  | Missing, expired, or malformed bearer token   |
| ERR_FORBIDDEN       | 403  | Valid token without the required scope        |
| ERR_NOT_FOUND       | 404  | Route or resource does not exist              |
| ERR_TOKEN_REPLAY    | 409  | A spent refresh token was presented again     |
| ERR_PAYLOAD_TOO_BIG | 413  | Body exceeded the 10 MB limit                 |
| ERR_RATE_LIMITED    | 429  | Per-key rate limit exceeded                   |
| ERR_UPSTREAM_TIMEOUT| 504  | Backend did not respond within the timeout    |

## Timeouts and retries

The default upstream timeout is **30 seconds**. It can be lowered per route with
the `timeout_ms` field in the route definition, but it cannot be raised above 30
seconds without a platform exception.

The gateway retries idempotent methods (GET, HEAD, PUT, DELETE) up to 2 times
with exponential backoff starting at 100 ms. POST is never retried
automatically, because the gateway cannot know whether the request was applied.
Send an `Idempotency-Key` header if you want POST retries; keys are remembered
for 24 hours.

## Request size limits

Request bodies are capped at 10 MB and response bodies at 50 MB. File uploads
larger than 10 MB must use the presigned-URL flow via `/v1/uploads/presign`
rather than posting through the gateway.
