"""Backend-agnostic generation smoke check.

    python eval/check_llm.py --list              # what is reachable right now
    python eval/check_llm.py                     # auto-select and call it
    python eval/check_llm.py --backend ollama --model llama3.1:8b
    python eval/check_llm.py --backend openai    # llama.cpp / LM Studio / vLLM

Exits non-zero on failure so it can gate the pipeline, exactly like the Qdrant
check. Nothing else in the project needs an API key — only this stage.
"""

import os
import sys
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

from generation.llm import available_backends, get_llm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

PROMPT = "Reply with exactly the words: this is a test."


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=None,
                        choices=("anthropic", "ollama", "openai", "stub", "auto"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--list", action="store_true",
                        help="probe every backend and exit")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args(argv)

    if args.list:
        print("Backend availability:")
        for name, ok in available_backends().items():
            print(f"  {name:10s} {'reachable' if ok else 'not reachable'}")
        print("\nLocal options if none are reachable:")
        print("  ollama serve && ollama pull llama3.1:8b   -> RAG_LLM_BACKEND=ollama")
        print("  llama-server -m model.gguf --port 8080    -> RAG_LLM_BACKEND=openai")
        return 0

    try:
        llm = get_llm(args.backend, model=args.model)
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Backend: %s | model: %s", llm.name, llm.model)

    try:
        response = llm.complete(
            PROMPT,
            system="You are a terse assistant.",
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        logger.error("Generation failed on the %s backend: %s", llm.name, exc)
        return 1

    logger.info("Response: %s", response.text)
    logger.info("Tokens: %d in / %d out", response.input_tokens, response.output_tokens)
    logger.info("Latency: %.2fs | cost: $%.6f", response.latency_s, response.cost_usd)

    if not response.text.strip():
        logger.error("Backend returned an empty response.")
        return 1

    logger.info("Success!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
