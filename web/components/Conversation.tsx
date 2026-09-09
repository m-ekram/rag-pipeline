"use client";

import { useEffect, useRef } from "react";
import type { ChatTurn, Citation, AnswerMetrics } from "@/rag/types";

const SAMPLES = [
  "What is this document about?",
  "Summarise the key findings",
  "मतदान केंद्र कहाँ है?",
];

export function Conversation({
  turns,
  ready,
  onSample,
}: {
  turns: ChatTurn[];
  ready: boolean;
  onSample: (q: string) => void;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const lastLength = useRef(0);

  useEffect(() => {
    // Follow the stream, but only while it is actually growing — otherwise
    // every re-render yanks the reader back down mid-scroll.
    const total = turns.reduce((n, t) => n + t.answer.length, 0);
    if (total !== lastLength.current) {
      lastLength.current = total;
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [turns]);

  if (turns.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <div className="max-w-lg text-center">
          <h2 className="font-serif text-[32px] leading-tight tracking-tight">
            Ask a folder of documents
          </h2>
          <p className="mx-auto mt-3 max-w-[46ch] text-[15px] leading-relaxed text-muted">
            Scanned pages are read with OCR, chunked and indexed locally. Every
            answer cites the pages it came from — and when the folder cannot
            answer, it says so instead of inventing something.
          </p>
          {ready && (
            <div className="mt-7 flex flex-wrap justify-center gap-2">
              {SAMPLES.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => onSample(q)}
                  lang={/[ऀ-ॿ]/.test(q) ? "hi" : "en"}
                  className="rounded-full border border-rule bg-surface px-3.5 py-1.5 text-[13px] text-ink-soft transition hover:border-accent hover:text-accent"
                >
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-9 px-6 py-8">
      {turns.map((turn) => (
        <Turn key={turn.id} turn={turn} />
      ))}
      <div ref={endRef} />
    </div>
  );
}

function Turn({ turn }: { turn: ChatTurn }) {
  const hindi = /[ऀ-ॿ]/;

  return (
    <article className="rise flex flex-col gap-4">
      <div className="flex justify-end">
        <p
          lang={hindi.test(turn.question) ? "hi" : undefined}
          className="max-w-[85%] rounded-2xl rounded-br-md bg-accent-wash px-4 py-2.5 text-[15px] whitespace-pre-wrap"
        >
          {turn.question}
        </p>
      </div>

      {turn.error ? (
        <Banner tone="danger" title="Request failed">
          <code className="font-mono text-[12px] break-words">{turn.error}</code>
        </Banner>
      ) : turn.abstained ? (
        <Banner tone="abstain" title="No answer from this folder">
          <p className="text-[13px] leading-relaxed opacity-90">
            {turn.reason ??
              "The retrieved evidence did not support an answer, so the system declined rather than guessing."}
          </p>
        </Banner>
      ) : (
        <div className="flex flex-col gap-4">
          <div
            lang={hindi.test(turn.answer) ? "hi" : undefined}
            className={`text-[15.5px] leading-[1.7] whitespace-pre-wrap ${
              turn.streaming && !turn.answer ? "caret text-muted" : ""
            }`}
          >
            {turn.answer}
            {turn.streaming && turn.answer ? <span className="caret" /> : null}
          </div>

          {turn.citations && turn.citations.length > 0 && (
            <Provenance citations={turn.citations} />
          )}
          {turn.metrics && <Metrics metrics={turn.metrics} />}
        </div>
      )}
    </article>
  );
}

/**
 * Citations render as a registry strip rather than footnotes: this corpus is a
 * page-numbered government record, so the page a claim came from is the thing
 * a reader checks first.
 */
function Provenance({ citations }: { citations: Citation[] }) {
  return (
    <details className="group rounded-xl border border-rule-soft bg-surface">
      <summary className="flex cursor-pointer items-center gap-2 px-4 py-2.5 font-mono text-[11px] tracking-[0.08em] text-muted uppercase select-none marker:content-none">
        <span className="transition group-open:rotate-90">›</span>
        {citations.length} source{citations.length > 1 ? "s" : ""}
      </summary>
      <div className="flex flex-col gap-3 border-t border-rule-soft px-4 py-3.5">
        {citations.map((c) => (
          <div key={c.n} className="flex gap-3">
            <span className="mt-0.5 shrink-0 rounded-md bg-accent-wash px-1.5 py-0.5 font-mono text-[11px] font-medium text-accent">
              {c.n}
            </span>
            <div className="min-w-0 flex-1">
              <p className="font-mono text-[11px] break-words text-ink-soft">
                {c.citation}
                {c.source ? ` · ${c.source}` : ""}
              </p>
              <p
                lang={/[ऀ-ॿ]/.test(c.preview) ? "hi" : undefined}
                className="mt-1 line-clamp-3 text-[12.5px] leading-relaxed text-muted"
              >
                {c.preview}
              </p>
            </div>
          </div>
        ))}
      </div>
    </details>
  );
}

function Metrics({ metrics }: { metrics: AnswerMetrics }) {
  const items: [string, string][] = [
    ["evidence", `${metrics.n_evidence_used}/${metrics.n_candidates}`],
    ["grounded", metrics.grounded ? "yes" : "no"],
    ["tokens", `${metrics.input_tokens}→${metrics.output_tokens}`],
  ];
  if (metrics.model) items.push(["model", metrics.model]);
  if (metrics.latency_generation_ms)
    items.push(["gen", `${(metrics.latency_generation_ms / 1000).toFixed(1)}s`]);
  if (metrics.cost_usd) items.push(["cost", `$${metrics.cost_usd.toFixed(5)}`]);

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10.5px] text-faint">
      {items.map(([k, v]) => (
        <span key={k}>
          {k} <b className="font-medium text-muted">{v}</b>
        </span>
      ))}
    </div>
  );
}

function Banner({
  tone,
  title,
  children,
}: {
  tone: "abstain" | "danger";
  title: string;
  children: React.ReactNode;
}) {
  const skin =
    tone === "abstain"
      ? "bg-abstain-wash text-abstain"
      : "bg-danger-wash text-danger";
  return (
    <div className={`flex flex-col gap-1.5 rounded-xl px-4 py-3.5 ${skin}`}>
      <b className="text-[14px] font-semibold">{title}</b>
      {children}
    </div>
  );
}
