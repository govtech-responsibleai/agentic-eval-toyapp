# Model-call limits and backpressure

Overload is refused before a turn starts, and model calls inside an admitted
turn wait for a slot but are never rejected. Three environment variables set
the limits; the defaults suit a key with 200 requests per minute (RPM) and 2M
tokens per minute (TPM) that only this app uses.

| Variable | Default | Effect |
|---|---|---|
| `TOYAPP_MAX_PENDING_TURNS` | `128` | Turns admitted at once. The next message request gets `503` with code `server_busy` and `Retry-After: 10`. It runs no agent code, creates no trace and stores no turn. |
| `TOYAPP_LLM_MAX_REQUESTS_PER_MINUTE` | `180` | Model-call starts are spaced evenly at this rate across the whole process. |
| `TOYAPP_LLM_MAX_CONCURRENT_CALLS` | `16` | Most model calls in flight at once; a guard against slow calls. |

Each value must be at least 1. Time spent waiting for a slot does not count
against `TOYAPP_GENERATION_TIMEOUT_SECONDS`, and the OpenAI client keeps
`max_retries=0` so a real provider `429` stays visible as `rate_limited`. There
is no token limiter. `server_busy` and `agent_unavailable` are both `503`;
tell them apart by the `code` field.

A turn makes several model calls: a policy question about three (coordinator,
RAG search step, RAG answer step) and a report analysis two.

| Traffic | Binding limit | Turns per minute at 200 RPM / 2M TPM |
|---|---|---|
| Policy question | requests | about 60 |
| Report analysis, small file | requests | up to 90 |
| Report analysis, PDF at the size cap | tokens | about 24 |

Token figures are estimates at 4 characters per token, not measurements.

## Tuning

Use this when the key's limits change.

1. **Request rate.** Set `TOYAPP_LLM_MAX_REQUESTS_PER_MINUTE` to
   `floor(0.9 x key RPM)`. The margin covers sliding-window counting and clock
   differences. If another tool shares the key, subtract its usage from the RPM
   first. Do not go above 90%: the cost of a `429` is a failed turn.
2. **Token check.** Confirm `turns per minute x tokens per turn <= 0.9 x key
   TPM`, taking tokens per turn from Langfuse generation usage or the provider
   dashboard. If it fails for your traffic (typically large PDFs), lower the
   request rate until it holds.
3. **Concurrency guard.** Set `TOYAPP_LLM_MAX_CONCURRENT_CALLS` to about
   `1.5 x (request rate per second) x (mean call latency in seconds)`. This is
   Little's law with 50% headroom. Get the latency from Langfuse generation
   durations, or time a few turns serially and divide by `model_calls` in each
   response. Report analysis carries a much larger prompt and can run tens of
   seconds, so a 30-second call needs ten times the slots of a 3-second call;
   its latency has not been measured. Too low shows as a low call rate with the
   provider idle; too high is harmless while pacing is correct.
4. **Admission cap.** Set `TOYAPP_MAX_PENDING_TURNS` to about
   `(request rate per minute / calls per turn) x acceptable wait in minutes`.
   The cap bounds queue time; it does not add capacity.

Worked examples, assuming 3-second calls, 3 calls per policy turn and a
2-minute acceptable wait (replace with measured values):

| Key RPM | Request rate | Concurrency guard | Policy turns per minute | Admission cap |
|---|---|---|---|---|
| 200 | 180 | 16 | 60 | 128 |
| 500 | 450 | 36 | 150 | 300 |
| 1000 | 900 | 68 | 300 | 600 |

At about one policy turn per second, 100 simultaneous turns take on the order
of 100 seconds to drain. The default cap accepts a 100-request burst but its
last request waits about that long, so clients need longer timeouts or a
smaller cap.

## Verify after changing anything

Run a controlled burst (for example 100 concurrent turns on distinct sessions)
against the live provider. It spends real quota, so it is never part of
`make ci`. Then read the signals, changing one setting at a time:

| Signal | Meaning | Action |
|---|---|---|
| Logs show a provider failure with code `rate_limited` | The provider refused a call | Lower the request rate; check for other users of the key; recheck the token budget |
| Responses with `server_busy` | Admission cap reached | Expected beyond capacity; raise the cap only if clients can wait longer |
| Slow turns, no failures, dashboard well below limits | Settings too conservative | Raise the request rate toward 90% of RPM, then the concurrency guard |
| Call rate far below the request rate setting | Concurrency guard too low | Raise it per step 3 |

## Limits that do not move with the key

- Retrieval stays single-slot and handles roughly 190 searches per second.
- The admission count, pacing clock and semaphore live in one process. Keep a
  single worker; with several, each enforces its own limits and the provider
  sees their sum.
- A provider may cap simultaneous requests separately from RPM and TPM. If so,
  keep the concurrency guard at or below that limit.
- A client that disconnects mid-turn still holds its admission slot until the
  turn finishes.
