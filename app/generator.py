"""Local text generation on ONNX Runtime (onnxruntime-genai).

Used on the retrieval side only - never to write answers:
  - doc2query: questions each chunk answers, indexed with it (app/doc2query.py)
  - query rewriting: paraphrases and a hypothetical passage per question
    (app/query_rewrite.py)

Default model: Phi-3.5-mini-instruct, int4 for CPU (~2.7 GB):

    HF_HUB_DISABLE_XET=1 huggingface-cli download microsoft/Phi-3.5-mini-instruct-onnx \
        --include "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*" --local-dir models/phi-onnx

ONNX Runtime ships signed binaries, so this also runs where application
control blocks unsigned native wheels (llama.cpp, PyTorch).
"""

from __future__ import annotations

import functools
from pathlib import Path

import config


class LocalGenerator:
    """Greedy (deterministic) chat completion with a Phi-3-family model."""

    def __init__(self, model_dir: str):
        import onnxruntime_genai as og

        if not Path(model_dir).is_dir():
            raise SystemExit(
                f"Generation model not found at {Path(model_dir).resolve()}. Download it with:\n"
                "  HF_HUB_DISABLE_XET=1 huggingface-cli download microsoft/Phi-3.5-mini-instruct-onnx "
                '--include "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*" --local-dir models/phi-onnx'
            )
        self._og = og
        self.model_dir = model_dir
        self._model = og.Model(model_dir)
        self._tokenizer = og.Tokenizer(self._model)

    def chat(self, prompt: str, max_new_tokens: int = 128) -> str:
        # Phi-3 chat template.
        tokens = self._tokenizer.encode(f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n")
        params = self._og.GeneratorParams(self._model)
        params.set_search_options(max_length=len(tokens) + max_new_tokens, do_sample=False)
        generator = self._og.Generator(self._model, params)
        generator.append_tokens(tokens)
        while not generator.is_done():
            generator.generate_next_token()
        return self._tokenizer.decode(generator.get_sequence(0)[len(tokens) :]).strip()


@functools.lru_cache(maxsize=1)
def get_generator(model_dir: str | None = None) -> LocalGenerator:
    return LocalGenerator(model_dir or config.GEN_MODEL_DIR)
