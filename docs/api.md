# API Contract

## Chat SSE

`POST /api/v1/chat/message`

Required JSON fields:

- `client_message_id`: UUID v4 string. Required for idempotent retries.
- `message` or `text`: user message text.

Optional JSON fields:

- `session_id`
- `birth_details`
- `matchmaking_details`

Behavior:

- If the same `client_message_id` is retried within the idempotency window, the backend replays the cached SSE events instead of invoking the LLM again.
- If the JWT expires soon, the stream returns an SSE error event with `code="token_expiring"` and the client should refresh the token before retrying.
- If the client disconnects, the backend stops streaming and stores the partial assistant message for analytics.
- If an authenticated user has no birth details in core-service and asks for chart-specific guidance, the stream metadata includes `needs_birth_details: true`.

## Birth Details Invalidation

`POST /internal/invalidate-cache`

Headers:

- `X-Internal-API-Key: <INTERNAL_API_KEY>`

Body:

```json
{
  "user_id": "user-123"
}
```

This clears the cached birth details for the user after the main app or core-service updates them.

## Daily Insight

`POST /internal/daily-insight`

Headers:

- `X-Internal-API-Key: <INTERNAL_API_KEY>`

Body:

```json
{
  "birth_details": {
    "name": "Ada",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "birth_datetime": "1990-01-01T12:00:00+00:00",
    "timezone_str": "Etc/UTC",
    "ayanamsha": "LAHIRI",
    "house_system": "W"
  },
  "session_id": "session-123",
  "external_user_id": "user-123",
  "long_term_memory": "- last_concern: career delay",
  "preferred_language": "en"
}
```

Behavior:

- Generates a scheduler-ready daily insight payload using chart, transits, predictions, and optional memory context.
- If `long_term_memory` is omitted, the service tries to load it from stored memory using `external_user_id` or `session_id`.
- The response includes `focus_area`, `headline`, `message`, `action`, `push_text`, `pattern_narrative`, and `pattern_confidence`.
