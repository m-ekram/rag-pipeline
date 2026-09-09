# Sanchay — Next.js frontend

Chat interface for the RAG backend: pick a folder, pick an engine, ask.

## Run

Two processes. The backend first:

```bash
cd ..
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Then the frontend:

```bash
pnpm install
pnpm dev              # http://localhost:3000
```

If port 8000 is taken, point the proxy elsewhere:

```bash
RAG_API_URL=http://127.0.0.1:8100 pnpm dev
```

## How it talks to the backend

`next.config.mjs` rewrites `/api/*` to FastAPI, so the browser only ever sees
one origin — no CORS, and rewrites pass streaming bodies through unbuffered.

Both long operations stream **newline-delimited JSON over POST** rather than
SSE, because `EventSource` is GET-only and both need a request body. A network
chunk can split a line anywhere, so `rag/client.ts` buffers partial lines until
a newline actually arrives.

| Endpoint | Streams | Events |
|---|---|---|
| `GET /api/backends` | no | reachable engines + their models |
| `GET /api/browse` | no | directory listing with data-file counts |
| `POST /api/index` | yes | `progress` → `done` \| `error` |
| `POST /api/chat` | yes | `status` → `token`… → `done` \| `error` |

## Layout

```
app/        layout (fonts, theme bootstrap), globals.css (tokens), page.tsx (state)
components/ Sidebar, FolderPicker, Conversation
rag/        client.ts (NDJSON reader), types.ts
```

Named `rag/`, not `lib/` — the repo's `.gitignore` had an unanchored `lib/`
rule that would have silently excluded it. That rule is now root-anchored, but
the name stays as a reminder.

## Design notes

- **Theme** is applied by an inline script before paint, so the first frame is
  never the wrong ground colour. Three states: explicit light, explicit dark,
  and unset (follows the OS).
- **Devanagari** is set via `:lang(hi)` with Noto Sans Devanagari loaded up
  front. System fallbacks render Hindi differently on every machine, and the
  corpus is Devanagari.
- **Abstention is a designed state**, not an error toast — it is the behaviour
  the whole project exists to demonstrate, so it gets its own amber panel
  separate from the red failure panel.
- **Citations render as a provenance strip** with page numbers in monospace:
  for a page-numbered government record, the page a claim came from is the
  first thing a reader checks.
