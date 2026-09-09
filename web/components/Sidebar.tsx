"use client";

import type { Backend } from "@/rag/types";

export interface IndexState {
  running: boolean;
  log: { text: string; error?: boolean }[];
  percent: number;
  ready: { documents: number; files: number; backend: string; model: string } | null;
}

export function Sidebar({
  folder,
  indexable,
  backends,
  backendId,
  model,
  ocrLang,
  index,
  theme,
  onOpenPicker,
  onBackend,
  onModel,
  onOcrLang,
  onIndex,
  onToggleTheme,
}: {
  folder: string | null;
  indexable: number;
  backends: Backend[];
  backendId: string;
  model: string;
  ocrLang: string;
  index: IndexState;
  theme: "light" | "dark" | null;
  onOpenPicker: () => void;
  onBackend: (id: string) => void;
  onModel: (m: string) => void;
  onOcrLang: (l: string) => void;
  onIndex: () => void;
  onToggleTheme: () => void;
}) {
  const backend = backends.find((b) => b.id === backendId);

  return (
    <aside className="flex h-full min-h-0 flex-col gap-6 overflow-y-auto border-r border-rule bg-surface px-5 py-5">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="font-serif text-[26px] leading-none tracking-tight">Sanchay</h1>
          <p className="mt-1 font-mono text-[10.5px] tracking-[0.14em] text-faint uppercase">
            संचय · archive retrieval
          </p>
        </div>
        <button
          type="button"
          onClick={onToggleTheme}
          aria-label="Toggle colour theme"
          className="rounded-lg border border-rule p-1.5 text-muted transition hover:border-accent hover:text-accent"
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </header>

      <Field label="Data folder">
        <button
          type="button"
          onClick={onOpenPicker}
          className="flex w-full items-center gap-2.5 rounded-xl border border-rule bg-sunk px-3 py-2.5 text-left transition hover:border-accent"
        >
          <span className="text-base leading-none">📁</span>
          <span
            className={`min-w-0 flex-1 truncate ${
              folder ? "font-mono text-[11.5px] text-ink" : "text-[13.5px] text-muted"
            }`}
            title={folder ?? undefined}
          >
            {folder ?? "Choose a folder…"}
          </span>
          <span className="text-faint">▾</span>
        </button>
        <Status
          tone={folder ? (indexable ? "good" : "bad") : "idle"}
          text={
            folder
              ? indexable
                ? `${indexable} document${indexable > 1 ? "s" : ""} ready`
                : "No documents in this folder"
              : "No folder selected"
          }
        />
      </Field>

      <Field label="Engine">
        <select
          value={backendId}
          onChange={(e) => onBackend(e.target.value)}
          className="w-full rounded-xl border border-rule bg-sunk px-3 py-2.5 text-[13.5px] outline-none transition focus:border-accent"
        >
          {backends.map((b) => (
            <option key={b.id} value={b.id} disabled={!b.available}>
              {b.available ? b.label : `${b.label} — unavailable`}
            </option>
          ))}
        </select>

        <select
          value={model}
          onChange={(e) => onModel(e.target.value)}
          disabled={!backend?.models.length}
          className="w-full rounded-xl border border-rule bg-sunk px-3 py-2.5 text-[13.5px] outline-none transition focus:border-accent disabled:opacity-50"
        >
          {backend?.models.length ? (
            backend.models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))
          ) : (
            <option value="">
              {backend?.available ? "server default" : "—"}
            </option>
          )}
        </select>

        <Status
          tone={backend?.available ? "good" : "bad"}
          text={
            backend?.available
              ? backend.models.length
                ? `${backend.models.length} model${backend.models.length > 1 ? "s" : ""} available`
                : "reachable"
              : backend?.hint || "not reachable"
          }
        />
      </Field>

      <Field label="OCR language">
        <select
          value={ocrLang}
          onChange={(e) => onOcrLang(e.target.value)}
          className="w-full rounded-xl border border-rule bg-sunk px-3 py-2.5 text-[13.5px] outline-none transition focus:border-accent"
        >
          <option value="en">English</option>
          <option value="hi">Hindi · देवनागरी</option>
          <option value="urd">Urdu · اردو</option>
        </select>
      </Field>

      <button
        type="button"
        onClick={onIndex}
        disabled={!folder || !indexable || index.running}
        className="relative w-full overflow-hidden rounded-xl bg-accent px-4 py-3 text-[14.5px] font-semibold text-on-accent transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
      >
        {index.running ? "Indexing…" : "Index folder"}
      </button>

      {index.running || index.log.length > 0 ? (
        <div className="flex flex-col gap-2.5">
          <div className="relative h-1 overflow-hidden rounded-full bg-sunk">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-300"
              style={{ width: `${index.percent}%` }}
            />
          </div>
          <div className="flex max-h-40 flex-col gap-1 overflow-y-auto rounded-lg bg-sunk px-3 py-2.5 font-mono text-[10.5px] leading-relaxed">
            {index.log.map((line, i) => (
              <p
                key={i}
                className={`break-words ${line.error ? "text-danger" : "text-muted"}`}
              >
                {line.text}
              </p>
            ))}
          </div>
        </div>
      ) : null}

      {index.ready && (
        <div className="rise flex flex-col gap-1 rounded-xl bg-grounded-wash px-3.5 py-3 text-grounded">
          <b className="text-[13px] font-semibold">
            {index.ready.documents} unit{index.ready.documents === 1 ? "" : "s"} from{" "}
            {index.ready.files} file{index.ready.files === 1 ? "" : "s"}
          </b>
          <span className="font-mono text-[11px] opacity-85">
            {index.ready.backend} · {index.ready.model || "default"}
          </span>
        </div>
      )}

      <footer className="mt-auto border-t border-rule-soft pt-4 text-[11.5px] leading-relaxed text-faint">
        Runs entirely on <code className="font-mono">127.0.0.1</code>. Answers cite
        the pages they came from, and the system abstains rather than guessing when
        the folder holds no evidence.
      </footer>
    </aside>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <label className="font-mono text-[10.5px] font-medium tracking-[0.11em] text-faint uppercase">
        {label}
      </label>
      {children}
    </div>
  );
}

function Status({ tone, text }: { tone: "good" | "bad" | "idle"; text: string }) {
  const dot =
    tone === "good" ? "bg-grounded" : tone === "bad" ? "bg-danger" : "bg-faint";
  return (
    <p className="flex items-center gap-2 text-[12px] text-muted">
      <span className={`size-1.5 shrink-0 rounded-full ${dot}`} />
      <span className="min-w-0 truncate">{text}</span>
    </p>
  );
}

function MoonIcon() {
  return (
    <svg className="size-4" viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg className="size-4" viewBox="0 0 16 16" fill="none" aria-hidden>
      <circle cx="8" cy="8" r="3.1" stroke="currentColor" strokeWidth="1.2" />
      <path
        d="M8 1v1.6M8 13.4V15M15 8h-1.6M2.6 8H1m11-5-1.1 1.1M5.1 10.9 4 12m8 0-1.1-1.1M5.1 5.1 4 4"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
      />
    </svg>
  );
}
