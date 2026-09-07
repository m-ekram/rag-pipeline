"""Phase 0 smoke check: confirm the Anthropic API key works end to end.

Run with `python eval/check_anthropic.py`. Exits non-zero on failure so it can
gate the rest of the pipeline.
"""

import os
import sys
import logging

import anthropic
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

# Overridable so the generation phase can A/B models without editing code.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")


def check_api():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or export it."
        )

    client = anthropic.Anthropic()

    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude. Say this is a test.",
            }
        ],
    )

    # response.content is a list of blocks (text, thinking, ...) — filter by type
    # rather than assuming content[0] is text.
    text = "".join(block.text for block in message.content if block.type == "text")

    logger.info("Model: %s", message.model)
    logger.info("Response: %s", text.strip())
    logger.info(
        "Tokens: %d in / %d out",
        message.usage.input_tokens,
        message.usage.output_tokens,
    )
    logger.info("Success!")


if __name__ == "__main__":
    try:
        check_api()
    except anthropic.AuthenticationError:
        logger.error("Invalid ANTHROPIC_API_KEY.")
        sys.exit(1)
    except anthropic.NotFoundError:
        logger.error("Unknown model %r.", MODEL)
        sys.exit(1)
    except anthropic.RateLimitError as exc:
        logger.error("Rate limited: %s", exc)
        sys.exit(1)
    except anthropic.APIStatusError as exc:
        logger.error("API error (%s): %s", exc.status_code, exc.message)
        sys.exit(1)
    except anthropic.APIConnectionError as exc:
        logger.error("Could not reach the Anthropic API: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Anthropic smoke check failed: %s", exc)
        sys.exit(1)
