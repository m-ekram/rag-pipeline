# 7. Limitations and Future Work

## 7.1 Limitations

**Uncalibrated abstention.** The threshold gate is in place, but its default
threshold (−2.0) never abstains once any evidence is retrieved: retrieval
scores come from three different scales — Reciprocal Rank Fusion scores near
0.03, sigmoid-normalised cross-encoder scores in (0, 1), and a fixed 1.0 for
exact structured lookups. A single threshold across these scales is not
meaningful, so the system currently relies on the model's own
`INSUFFICIENT_EVIDENCE` reply to decline. The project's research plan — a
risk–coverage analysis of abstention under corpus contamination — requires a
per-intent gate calibrated on held-out questions, which was out of scope
here.

**OCR on the test machine.** PaddlePaddle 3.3.1's accelerated CPU path
(oneDNN) failed on the Windows test laptop with an unimplemented-attribute
error; disabling the new executor did not avoid it, and without oneDNN a
scanned roll page took 104–118 seconds (about 50 minutes for one 28-page
roll). Downgrading to PaddlePaddle 3.0.0 was tried in isolation and rejected:
its bundled runtime conflicted with PyTorch's (`WinError 127` loading
`shm.dll`). Tesseract, which the pipeline already supports as its fallback
engine, needs a machine-wide installation that could not be completed on the
test laptop. The electoral rolls were therefore **not measured** in this
work. The roll-specific logic (record re-alignment, household grouping,
field and relation search) is covered by the automated tests on OCR text,
and the Linux deployment the OCR settings were originally tuned for does not
have this problem. The code now reports a missing OCR engine as an error
instead of producing blank pages.

**CPU-bound embedding.** The multilingual embedding model processes about
8–10 chunks per second on the test CPU. The first index of the 335-page
Master Plan (1,616 chunks) therefore takes about five minutes. Fingerprinted
reuse makes this a one-off cost per document set, but a very large folder
(for example all 435 electoral rolls) would take hours to embed on this
hardware.

**One request at a time.** All pipeline work runs on one thread. The
interface now reports when a question is waiting behind an index, but a user
cannot chat with one folder while another is being indexed.

**Measurement scope.** Latency was measured with one local model on one
laptop. The hosted engine (Groq) is implemented and tested with fakes but was
not configured, so no hosted-engine latencies are reported. Answer quality
was checked by inspection of the golden-question answers, not by a labelled
evaluation.

## 7.2 Future work

- **Calibrated, per-intent abstention**, fitted on a held-out split and
  reported as risk–coverage curves across contamination levels, as set out
  in the project plan.
- **Faster embedding**: an ONNX or 8-bit quantised export of the embedding
  model typically gives a two- to three-fold CPU speed-up, which would bring
  first-index times for the Master Plan down to one or two minutes.
- **Streaming indexing into a live session**, so questions can be asked about
  the first files of a large folder while the rest are still being processed.
- **A labelled evaluation set** for the three document families, so that
  retrieval changes (chunk sizes, reranker choice, evidence budgets) can be
  compared on recall and answer accuracy rather than latency alone.
- **Hosted-engine measurements** with Groq, to quantify the latency gap
  between local CPU generation and a hosted accelerator.
