export type BackendId = "ollama" | "openai" | "groq" | "anthropic" | "stub";

export interface Backend {
  id: BackendId;
  label: string;
  available: boolean;
  models: string[];
  hint: string;
  /** Model size in bytes, when the backend reports it (Ollama). */
  sizes?: Record<string, number>;
  /** The fastest backend that can answer on this machine. */
  recommended?: boolean;
}

export interface DirEntry {
  name: string;
  path: string;
  data_files: number;
  nested_files: number;
}

export interface FileEntry {
  name: string;
  size: number;
  suffix: string;
}

export interface BrowseResult {
  path: string;
  parent: string | null;
  dirs: DirEntry[];
  files: FileEntry[];
  indexable: number;
}

export interface Citation {
  n: number;
  citation: string;
  page: number | null;
  source: string;
  preview: string;
}

export interface AnswerMetrics {
  n_candidates: number;
  n_evidence_used: number;
  n_evidence_dropped: number;
  grounded: boolean;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  model: string | null;
  backend: string | null;
  latency_retrieval_ms?: number;
  latency_rerank_ms?: number;
  latency_generation_ms?: number;
}

export interface Health {
  ok: boolean;
  sessions: number;
  warm: { state: "cold" | "warming" | "ready" | "error"; seconds: number | null; detail: string };
}

/** Events the NDJSON endpoints emit. */
export type StreamEvent =
  | {
      type: "progress";
      stage: string;
      message: string;
      current?: number | null;
      total?: number | null;
    }
  | { type: "status"; stage?: string; message: string }
  /** Sent every few seconds while the backend is busy but has nothing new to say. */
  | { type: "heartbeat"; stage: string; elapsed: number }
  | { type: "token"; text: string }
  | {
      type: "done";
      session_id?: string;
      documents?: number;
      files?: number;
      folder?: string;
      backend?: string;
      model?: string;
      answer?: string;
      abstained?: boolean;
      decision?: string;
      reason?: string;
      grounded?: boolean;
      citations?: Citation[];
      metrics?: AnswerMetrics;
    }
  | { type: "error"; message: string; detail?: string };

export interface ChatTurn {
  id: string;
  question: string;
  answer: string;
  streaming: boolean;
  /** What the backend is doing right now, e.g. "Generating with groq · llama-3.1-8b-instant". */
  activity?: string;
  /** Seconds since the question was sent, while waiting. */
  elapsed?: number;
  abstained?: boolean;
  reason?: string;
  citations?: Citation[];
  metrics?: AnswerMetrics;
  error?: string;
}
