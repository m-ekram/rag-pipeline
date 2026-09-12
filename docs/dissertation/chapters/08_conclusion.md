# 8. Conclusion

This project set out to make a local RAG system answer questions over three
very different kinds of government and academic documents, through a web
interface, on an ordinary laptop. The pipeline already produced answers on
the command line; the web interface did not. The work therefore began as
diagnosis rather than as model tuning.

Tracing a single request from the browser to the model found that the most
visible failure — an interface that showed nothing — had nothing to do with
retrieval quality. A development proxy cut long requests off after 30
seconds and buffered the stream; the backend could not abandon work its
client had given up on; and the first request paid for library imports,
model loads and downloads. Streaming heartbeats, cancelling on disconnect,
warming models at startup and serving the interface from the API process
removed those failures, and made the remaining slow steps visible rather than
silent.

The optimisation work that followed was guided by measurement, and several of
its results were not the expected ones:

- **Embedding, not extraction, dominates the first index** on a CPU (316 of
  317 seconds for the Master Plan). Fingerprinting chunk texts made repeat
  indexing about 45 times faster, which matters more to a user than speeding
  up the first index.
- **Reranking cost depended on which model was chosen, not on the question.**
  Choosing the multilingual reranker by the share of Hindi text, rather than
  its mere presence, and capping its input cut retrieval from about three
  seconds to about one.
- **Shrinking the prompt was a poor trade.** On a CPU the model reads about
  29 prompt tokens per second, but cutting evidence by 30% saved only 7% of
  the wait and cost answers; the enclosing sections carry the facts a small
  model needs. The instructions, which the model server caches after the
  first question, were the part worth shortening.
- **A smaller model was faster but wrong** on a simple numeric question, so
  speed was not bought with correctness.

- **Short instructions change behaviour, not only speed.** A compact prompt
  made the 3B model reply to two table questions with a bare citation; one
  explicit rule ("answer in sentences that state the facts") restored them.

With the final configuration, a local 3-billion-parameter model answers with
a median time to the first token of 31.6 seconds and a complete answer in
38.5 seconds, every answer stating its fact with a citation. That remaining
cost is set by the hardware; the system already selects a hosted engine when
one is configured, and keeps the local model as an offline fallback.

The system now treats the three document families appropriately in one
folder — tables stay tables, prose keeps its section, voters stay one record
each — and every claim in this report is backed by a logged measurement
that can be reproduced with the scripts in the repository. The most important
open problem is the one the project's research plan identifies: a calibrated,
per-intent abstention gate, so that the system's decision to answer or to
decline is as trustworthy as its citations.
