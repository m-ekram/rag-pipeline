import type { Backend, BrowseResult, StreamEvent } from "./types";

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function fetchBackends(): Promise<{ backends: Backend[] }> {
  return getJSON("/api/backends");
}

export function browse(path?: string): Promise<BrowseResult> {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return getJSON(`/api/browse${query}`);
}

/**
 * Consume an NDJSON stream from a POST endpoint.
 *
 * EventSource only speaks GET, and both long operations here need a request
 * body, so the backend emits newline-delimited JSON and we read the response
 * body incrementally. A network chunk can split a line anywhere, so partial
 * lines are buffered until a newline actually arrives.
 */
export async function* streamNDJSON(
  url: string,
  body: unknown,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line) yield JSON.parse(line) as StreamEvent;
      }
    }
    const tail = buffer.trim();
    if (tail) yield JSON.parse(tail) as StreamEvent;
  } finally {
    // Releasing the lock lets an aborted request tear down instead of leaking
    // the reader when the user navigates away mid-answer.
    reader.releaseLock();
  }
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
