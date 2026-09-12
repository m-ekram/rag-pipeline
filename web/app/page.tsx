"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Conversation } from "@/components/Conversation";
import { FolderPicker } from "@/components/FolderPicker";
import { Sidebar, type IndexState } from "@/components/Sidebar";
import { fetchBackends, fetchHealth, streamNDJSON } from "@/rag/client";
import type { Backend, ChatTurn, Health } from "@/rag/types";

const EMPTY_INDEX: IndexState = { running: false, log: [], percent: 0, ready: null, activity: null };

const STAGE_LABELS: Record<string, string> = {
  queued: "Waiting for an earlier request",
  warming: "Loading models",
  starting: "Starting",
  scan: "Scanning folder",
  extract: "Extracting text",
  index: "Indexing",
  retrieval: "Searching",
  rerank: "Ranking evidence",
  generation: "Generating",
};

export default function Page() {
  const [backends, setBackends] = useState<Backend[]>([]);
  const [backendId, setBackendId] = useState("stub");
  const [model, setModel] = useState("");
  const [ocrLang, setOcrLang] = useState("auto");
  const [warm, setWarm] = useState<Health["warm"] | null>(null);

  const [folder, setFolder] = useState<string | null>(null);
  const [indexable, setIndexable] = useState(0);
  const [pickerOpen, setPickerOpen] = useState(false);

  const [index, setIndex] = useState<IndexState>(EMPTY_INDEX);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [asking, setAsking] = useState(false);
  const [draft, setDraft] = useState("");
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  /* ---------------------------------------------------------------- setup */

  useEffect(() => {
    setTheme(
      (localStorage.getItem("theme") as "light" | "dark" | null) ??
        (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"),
    );

    void (async () => {
      try {
        const { backends } = await fetchBackends();
        setBackends(backends);
        // Prefer the engine the server recommends: the fastest that can answer
        // on this machine. The stub exists to test retrieval, so it must never
        // be selected by default and pass fake answers off as real.
        const first =
          backends.find((b) => b.recommended) ??
          backends.find((b) => b.available && b.id !== "stub") ??
          backends.find((b) => b.available);
        if (first) {
          setBackendId(first.id);
          setModel(first.models[0] ?? "");
        }
      } catch {
        /* the sidebar shows every backend as unreachable */
      }
    })();
  }, []);

  // The backend loads its models in the background at startup. Showing that
  // state is what stops the first click from looking like a hang.
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let polls = 0;
    const poll = async () => {
      try {
        const health = await fetchHealth();
        if (stopped) return;
        setWarm(health.warm);
        if (health.warm.state === "ready" || health.warm.state === "error") return;
      } catch {
        /* backend not up yet */
      }
      if (!stopped && ++polls < 200) timer = setTimeout(poll, 3000);
    };
    void poll();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  };

  const selectBackend = (id: string) => {
    setBackendId(id);
    setModel(backends.find((b) => b.id === id)?.models[0] ?? "");
  };

  /* -------------------------------------------------------------- indexing */

  const runIndex = useCallback(async () => {
    if (!folder) return;
    setIndex({ ...EMPTY_INDEX, running: true });
    setSessionId(null);

    const log = (text: string, error = false) =>
      setIndex((s) => ({ ...s, log: [...s.log, { text, error }] }));

    try {
      for await (const ev of streamNDJSON("/api/index", {
        folder,
        backend: backendId,
        model: model || null,
        ocr_lang: ocrLang,
      })) {
        if (ev.type === "progress") {
          log(ev.message);
          setIndex((s) => ({
            ...s,
            activity: null,
            // Held below 100 until "done": the index stage can still take a while.
            percent:
              ev.total && ev.current != null
                ? Math.min(95, Math.round((100 * ev.current) / ev.total))
                : s.percent,
          }));
        } else if (ev.type === "status") {
          log(ev.message);
        } else if (ev.type === "heartbeat") {
          setIndex((s) => ({
            ...s,
            activity: `${STAGE_LABELS[ev.stage] ?? ev.stage} · ${Math.round(ev.elapsed)}s`,
          }));
        } else if (ev.type === "done") {
          setSessionId(ev.session_id ?? null);
          setIndex((s) => ({
            ...s,
            percent: 100,
            activity: null,
            ready: {
              documents: ev.documents ?? 0,
              files: ev.files ?? 0,
              backend: ev.backend ?? backendId,
              model: ev.model ?? model,
            },
          }));
          setTimeout(() => inputRef.current?.focus(), 60);
        } else if (ev.type === "error") {
          log(ev.message, true);
        }
      }
    } catch (err) {
      log(err instanceof Error ? err.message : String(err), true);
    } finally {
      setIndex((s) => ({ ...s, running: false, activity: null }));
    }
  }, [folder, backendId, model, ocrLang]);

  /* ------------------------------------------------------------------ chat */

  const ask = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || asking || !sessionId) return;

      const id = crypto.randomUUID();
      setTurns((t) => [...t, { id, question: q, answer: "", streaming: true }]);
      setDraft("");
      setAsking(true);

      const patch = (fields: Partial<ChatTurn>) =>
        setTurns((t) => t.map((x) => (x.id === id ? { ...x, ...fields } : x)));

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        for await (const ev of streamNDJSON(
          "/api/chat",
          { session_id: sessionId, question: q },
          controller.signal,
        )) {
          if (ev.type === "token") {
            setTurns((t) =>
              t.map((x) => (x.id === id ? { ...x, answer: x.answer + ev.text } : x)),
            );
          } else if (ev.type === "status") {
            patch({ activity: ev.message });
          } else if (ev.type === "heartbeat") {
            patch({ elapsed: ev.elapsed });
          } else if (ev.type === "done") {
            patch({
              // The streamed text carries raw [1] markers; the final answer has
              // them expanded into full citations, so prefer it when present.
              answer: ev.answer || "",
              streaming: false,
              abstained: ev.abstained,
              reason: ev.reason,
              citations: ev.citations,
              metrics: ev.metrics,
            });
          } else if (ev.type === "error") {
            patch({ streaming: false, error: ev.message });
          }
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          patch({
            streaming: false,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      } finally {
        setAsking(false);
        abortRef.current = null;
        patch({ streaming: false });
      }
    },
    [asking, sessionId],
  );

  // Aborting on unmount closes the stream, which the backend treats as a
  // cancellation and stops generating.
  useEffect(() => () => abortRef.current?.abort(), []);

  const canChat = Boolean(sessionId) && !asking;

  return (
    <div className="grid h-dvh grid-cols-1 overflow-hidden md:grid-cols-[320px_1fr]">
      <div className="hidden min-h-0 md:block">
        <Sidebar
          folder={folder}
          indexable={indexable}
          backends={backends}
          backendId={backendId}
          model={model}
          ocrLang={ocrLang}
          index={index}
          warm={warm}
          theme={theme}
          onOpenPicker={() => setPickerOpen(true)}
          onBackend={selectBackend}
          onModel={setModel}
          onOcrLang={setOcrLang}
          onIndex={() => void runIndex()}
          onToggleTheme={toggleTheme}
        />
      </div>

      <main className="flex min-h-0 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto">
          <Conversation turns={turns} ready={Boolean(sessionId)} onSample={ask} />
        </div>

        <div className="border-t border-rule bg-surface px-6 py-4">
          <div className="mx-auto flex w-full max-w-3xl items-end gap-2.5">
            <textarea
              ref={inputRef}
              value={draft}
              rows={1}
              disabled={!canChat}
              placeholder={
                sessionId ? "Ask something about this folder…" : "Index a folder to begin…"
              }
              onChange={(e) => {
                setDraft(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 190)}px`;
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  // Read the live DOM value rather than the `draft` closure: a
                  // fast typist (or an automated one) can land Enter before
                  // React flushes the last keystroke, and the closure would
                  // then submit stale — or empty — text.
                  void ask(e.currentTarget.value);
                }
              }}
              className="max-h-48 min-h-[46px] flex-1 resize-none rounded-xl border border-rule bg-sunk px-3.5 py-3 text-[15px] outline-none transition focus:border-accent disabled:opacity-55"
            />
            <button
              type="button"
              onClick={() => void ask(draft)}
              disabled={!canChat || !draft.trim()}
              aria-label="Send"
              className="grid size-[46px] shrink-0 place-items-center rounded-xl bg-accent text-on-accent transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-35"
            >
              {asking ? <Spinner /> : <ArrowIcon />}
            </button>
          </div>
          <p className="mx-auto mt-2 max-w-3xl text-[11.5px] text-faint">
            Enter to send · Shift+Enter for a new line
          </p>
        </div>
      </main>

      <FolderPicker
        open={pickerOpen}
        initialPath={folder}
        onClose={() => setPickerOpen(false)}
        onChoose={(path, count) => {
          setFolder(path);
          setIndexable(count);
          setPickerOpen(false);
          // A new folder invalidates the previous index, so drop the session
          // rather than letting questions answer from the old corpus.
          setSessionId(null);
          setIndex(EMPTY_INDEX);
        }}
      />
    </div>
  );
}

function ArrowIcon() {
  return (
    <svg className="size-[18px]" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M8 13V3m0 0L3.6 7.4M8 3l4.4 4.4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Spinner() {
  return (
    <svg className="size-[18px] animate-spin" viewBox="0 0 16 16" fill="none" aria-hidden>
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeWidth="1.8" opacity=".25" />
      <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
