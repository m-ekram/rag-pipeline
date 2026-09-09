"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { browse, formatBytes } from "@/rag/client";
import type { BrowseResult } from "@/rag/types";

export function FolderPicker({
  open,
  initialPath,
  onClose,
  onChoose,
}: {
  open: boolean;
  initialPath: string | null;
  onClose: () => void;
  onChoose: (path: string, indexable: number) => void;
}) {
  const [result, setResult] = useState<BrowseResult | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);

  const go = useCallback(async (path?: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await browse(path);
      setResult(data);
      setDraft(data.path);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) void go(initialPath ?? undefined);
  }, [open, initialPath, go]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    dialogRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const empty = result && result.dirs.length === 0 && result.files.length === 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4 backdrop-blur-[2px]"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="Select a data folder"
        className="rise flex max-h-[min(78vh,680px)] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-rule bg-surface shadow-[var(--shadow-lg)] outline-none"
      >
        <header className="flex flex-col gap-3 border-b border-rule px-5 py-4">
          <div className="flex items-baseline justify-between gap-4">
            <h2 className="font-serif text-xl">Select a data folder</h2>
            <span className="font-mono text-[11px] tracking-wider text-faint uppercase">
              pdf · txt · md · csv
            </span>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => result?.parent && go(result.parent)}
              disabled={!result?.parent}
              className="shrink-0 rounded-lg border border-rule bg-sunk px-3 py-2 text-sm text-ink-soft transition hover:border-accent hover:text-accent disabled:cursor-not-allowed disabled:opacity-40"
            >
              ↑ Up
            </button>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void go(draft.trim());
                }
              }}
              spellCheck={false}
              aria-label="Folder path"
              className="min-w-0 flex-1 rounded-lg border border-rule bg-sunk px-3 py-2 font-mono text-[12.5px] text-ink outline-none transition focus:border-accent"
            />
            <button
              type="button"
              onClick={() => void go(draft.trim())}
              className="shrink-0 rounded-lg border border-rule bg-sunk px-3 py-2 text-sm text-ink-soft transition hover:border-accent hover:text-accent"
            >
              Go
            </button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto py-1.5">
          {loading && (
            <p className="px-5 py-8 text-center text-sm text-muted">Reading folder…</p>
          )}

          {error && !loading && (
            <p className="mx-5 my-4 rounded-lg bg-danger-wash px-4 py-3 text-sm text-danger">
              {error}
            </p>
          )}

          {!loading && !error && result && (
            <>
              {result.dirs.map((dir) => {
                // Direct counts are actionable; nested counts tell you to keep
                // going. Without both, a project root shows `data/` as empty
                // even when every PDF lives two levels inside it.
                const badge = dir.data_files
                  ? { text: `${dir.data_files} file${dir.data_files > 1 ? "s" : ""}`, tone: "text-grounded" }
                  : dir.nested_files
                    ? { text: `${dir.nested_files} inside`, tone: "text-faint" }
                    : null;

                return (
                  <button
                    key={dir.path}
                    type="button"
                    onClick={() => void go(dir.path)}
                    className="group flex w-full items-center gap-3 px-5 py-2 text-left transition hover:bg-sunk"
                  >
                    <FolderIcon className="size-4 shrink-0 text-faint transition group-hover:text-accent" />
                    <span className="min-w-0 flex-1 truncate text-[14px]">{dir.name}</span>
                    {badge && (
                      <span className={`font-mono text-[11px] ${badge.tone}`}>
                        {badge.text}
                      </span>
                    )}
                  </button>
                );
              })}

              {result.files.map((file) => (
                <div
                  key={file.name}
                  className="flex items-center gap-3 px-5 py-2 text-ink-soft"
                >
                  <FileIcon className="size-4 shrink-0 text-faint" />
                  <span className="min-w-0 flex-1 truncate text-[13.5px]">{file.name}</span>
                  <span className="font-mono text-[11px] text-faint">
                    {formatBytes(file.size)}
                  </span>
                </div>
              ))}

              {empty && (
                <p className="px-5 py-8 text-center text-sm text-muted">
                  Nothing here that can be indexed.
                </p>
              )}
            </>
          )}
        </div>

        <footer className="flex items-center justify-between gap-4 border-t border-rule px-5 py-3.5">
          <span className="text-[13px] text-muted">
            {result?.indexable
              ? `${result.indexable} indexable file${result.indexable > 1 ? "s" : ""} here`
              : "No documents here — open a sub-folder"}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-rule bg-sunk px-4 py-2 text-sm text-ink-soft transition hover:border-accent hover:text-accent"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!result?.indexable}
              onClick={() => result && onChoose(result.path, result.indexable)}
              title={result?.indexable ? "" : "This folder has no indexable documents"}
              className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-on-accent transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
            >
              Use this folder
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

function FolderIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M1.5 4.2c0-.66.54-1.2 1.2-1.2h3.1c.4 0 .77.2.99.53l.62.94h5.99c.66 0 1.2.54 1.2 1.2v6.13c0 .66-.54 1.2-1.2 1.2H2.7c-.66 0-1.2-.54-1.2-1.2V4.2Z"
        stroke="currentColor"
        strokeWidth="1.2"
      />
    </svg>
  );
}

function FileIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" aria-hidden>
      <path
        d="M3.5 2.5h5.1L12.5 6.4v7.1a1 1 0 0 1-1 1h-8a1 1 0 0 1-1-1v-10a1 1 0 0 1 1-1Z"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <path d="M8.5 2.6V6.5h3.9" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}
